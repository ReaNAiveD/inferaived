use bytemuck;
use half::bf16;
use safetensors::tensor::TensorView;
use tokenizers::Encoding;
use wgpu::{
    BindGroupDescriptor, BindGroupLayout, BindGroupLayoutDescriptor, BindGroupLayoutEntry, Buffer, BufferDescriptor, CommandEncoderDescriptor, ComputePassDescriptor, ComputePipeline, ComputePipelineDescriptor, Device, PipelineCompilationOptions, PipelineLayoutDescriptor, Queue, util::{BufferInitDescriptor, DeviceExt}
};

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct GetRowsParams {
    source_offset: u32,        // in u32 elements
    source_row_stride: u32,    // in u32 elements (= hidden_size / 2)
    indices_offset: u32,       // in i32 elements
    output_offset: u32,        // in f32 elements
    output_row_stride: u32,    // in f32 elements

    hidden_size: u32,
    num_tokens: u32,
}

pub struct EmbeddingLookupWebgpu {
    hidden_size: usize,
    bind_group_layout: BindGroupLayout,
    pipeline: ComputePipeline,
    embeddings_buffer: wgpu::Buffer,
    uniform_buffer: wgpu::Buffer,
}

impl EmbeddingLookupWebgpu {
    /// Creates a new GPU-based embedding lookup.
    /// Note: `stride_src1` is in u32 units (hidden_size / 2) because the shader
    /// reads packed bf16 pairs as `array<u32>`.
    pub fn new<'data>(device: &Device, queue: &Queue, embeddings: TensorView<'data>, hidden_size: usize) -> Self {
        let shader_module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("embedding_lookup/shader"),
            source: wgpu::ShaderSource::Wgsl(std::borrow::Cow::Borrowed(include_str!(
                "wgsl-shaders/get_rows.wgsl"
            ))),
        });
        let src_bind_layout_entry = BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let idx_bind_layout_entry = BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: true },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let dst_bind_layout_entry = BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Storage { read_only: false },
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let uniform_bind_layout_entry = BindGroupLayoutEntry {
            binding: 3,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&BindGroupLayoutDescriptor {
            label: Some("embedding_lookup/bind_group_layout"),
            entries: &[
                src_bind_layout_entry,
                idx_bind_layout_entry,
                dst_bind_layout_entry,
                uniform_bind_layout_entry,
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&PipelineLayoutDescriptor {
            label: None,
            bind_group_layouts: &[&bind_group_layout],
            immediate_size: 0,
        });
        let pipeline = device.create_compute_pipeline(&ComputePipelineDescriptor {
            label: None,
            layout: Some(&pipeline_layout),
            module: &shader_module,
            entry_point: Some("main"),
            compilation_options: PipelineCompilationOptions::default(),
            cache: None,
        });

        let embeddings_buffer_size = embeddings.data().len() as wgpu::BufferAddress;
        let align_mask = wgpu::COPY_BUFFER_ALIGNMENT - 1;
        let padded_embedding_buffer_size =
            ((embeddings_buffer_size + align_mask) & !align_mask).max(wgpu::COPY_BUFFER_ALIGNMENT);
        let embeddings_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: padded_embedding_buffer_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let uniform_buffer = device.create_buffer(&BufferDescriptor {
            label: None,
            size: std::mem::size_of::<GetRowsParams>() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        queue.write_buffer(
            &embeddings_buffer,
            0,
            bytemuck::cast_slice(embeddings.data()),
        );
        Self {
            hidden_size,
            bind_group_layout,
            pipeline,
            embeddings_buffer,
            uniform_buffer,
        }
    }

    pub fn compute<'data>(&self, device: &Device, queue: &Queue, input_encoding: &[u32], dst_buffer: &Buffer) {
        let index_buffer = device.create_buffer_init(&BufferInitDescriptor {
            label: None,
            contents: bytemuck::cast_slice(input_encoding),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
        });
        // source_row_stride is in u32 units: each u32 packs two bf16 values,
        // so a row of hidden_size bf16 elements = hidden_size/2 u32s.
        let n_rows = input_encoding.len();
        queue.write_buffer(&self.uniform_buffer, 0u64, bytemuck::cast_slice(&[GetRowsParams {
            source_offset: 0,
            source_row_stride: (self.hidden_size / 2) as u32,
            indices_offset: 0,
            output_offset: 0,
            output_row_stride: self.hidden_size as u32,
            hidden_size: self.hidden_size as u32,
            num_tokens: n_rows as u32,
        }]));
        let mut command_encoder = device.create_command_encoder(&CommandEncoderDescriptor {
            label: Some("embedding_lookup/command_encoder"),
        });
        let mut compute_pass = command_encoder.begin_compute_pass(&ComputePassDescriptor {
            label: Some("embedding_lookup/compute_pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(&self.pipeline);
        let bind_group = device.create_bind_group(&BindGroupDescriptor {
            label: Some("embedding_lookup/bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: self.embeddings_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: index_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: dst_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: self.uniform_buffer.as_entire_binding(),
                },
            ],
        });
        compute_pass.set_bind_group(0, &bind_group, &[]);
        // Here we dispatch 256 threads per workgroup
        let workgroup_count = (n_rows as u32 + 255) / 256;
        compute_pass.dispatch_workgroups(workgroup_count, 1, 1);
        drop(compute_pass);
        queue.submit(Some(command_encoder.finish()));
    }
}

pub struct EmbeddingLookupCpu<'data> {
    hidden_size: usize,
    embed_tokens: TensorView<'data>
}

impl<'data> EmbeddingLookupCpu<'data> {
    pub fn new(embed_tokens: TensorView<'data>) -> Self {
        let hidden_size = embed_tokens.shape()[1];
        Self {
            hidden_size,
            embed_tokens,
        }
    }

    pub fn compute(&self, input_encoding: &[u32]) -> Vec<f32> {
        let row_width = self.hidden_size * std::mem::size_of::<bf16>();
        let embedding_row_num = self.embed_tokens.data().len() / row_width;
        input_encoding.iter().flat_map(|&idx| {
            let start = ((idx as usize) % embedding_row_num) * row_width;
            let end = start + row_width;
            let row_data = &self.embed_tokens.data()[start..end];
            // The safetensors store bf16 values as little-endian
            let row_floats: Vec<f32> = row_data
                .chunks_exact(2)
                .map(|chunk| {
                    let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
                    bf16::from_bits(bits).to_f32()
                })
                .collect();
            row_floats
        }).collect()
    }
}