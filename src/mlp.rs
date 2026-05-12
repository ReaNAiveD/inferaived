use safetensors::SafeTensors;

use crate::{log_tensor, mul_mat::MulMatWebgpu, silu_mul::SiluMulInplaceWebgpu};

pub struct MultiLayerPerceptron {
    mlp_gate_proj_mul_mat: MulMatWebgpu,
    mlp_gate_proj_buffer: wgpu::Buffer,
    mlp_up_proj_mul_mat: MulMatWebgpu,
    mlp_up_proj_buffer: wgpu::Buffer,
    mlp_silu_mul: SiluMulInplaceWebgpu,
    mlp_down_proj_mul_mat: MulMatWebgpu,
}

impl MultiLayerPerceptron {
    pub fn new<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        tensor: &SafeTensors<'data>,
        weight_prefix: &str,
        hidden_size: usize,
        seq_len: usize,
    ) -> Self {
        let mlp_gate_proj_weight_name = format!("{}.mlp.gate_proj.weight", weight_prefix);
        let mlp_gate_proj_weight = tensor.tensor(&mlp_gate_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_gate_proj_weight_name
        ));
        log_tensor(&mlp_gate_proj_weight_name, &mlp_gate_proj_weight);
        let mlp_gate_dim = mlp_gate_proj_weight.shape()[0] as usize;
        let mlp_gate_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_gate_proj_weight, hidden_size);
        let mlp_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_gate_proj_buffer"),
            size: (seq_len * mlp_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_up_proj_weight_name = format!("{}.mlp.up_proj.weight", weight_prefix);
        let mlp_up_proj_weight = tensor.tensor(&mlp_up_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_up_proj_weight_name
        ));
        log_tensor(&mlp_up_proj_weight_name, &mlp_up_proj_weight);
        let mlp_up_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_up_proj_weight, hidden_size);
        let mlp_up_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("linear_attention_layer/mlp_up_proj_buffer"),
            size: (seq_len * mlp_gate_dim * std::mem::size_of::<f32>()) as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_silu_mul = SiluMulInplaceWebgpu::new(&device, mlp_gate_dim);
        let mlp_down_proj_weight_name = format!("{}.mlp.down_proj.weight", weight_prefix);
        let mlp_down_proj_weight = tensor.tensor(&mlp_down_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_down_proj_weight_name
        ));
        log_tensor(&mlp_down_proj_weight_name, &mlp_down_proj_weight);
        let mlp_down_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_down_proj_weight, mlp_gate_dim);
        Self {
            mlp_gate_proj_mul_mat,
            mlp_gate_proj_buffer,
            mlp_up_proj_mul_mat,
            mlp_up_proj_buffer,
            mlp_silu_mul,
            mlp_down_proj_mul_mat,
        }
    }

    pub fn compute(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input_buffer: &wgpu::Buffer,
        output_buffer: &wgpu::Buffer,
        seq_len: usize,
    ) {
        self.mlp_gate_proj_mul_mat.compute(
            device,
            queue,
            input_buffer,
            &self.mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_up_proj_mul_mat.compute(
            device,
            queue,
            input_buffer,
            &self.mlp_up_proj_buffer,
            seq_len,
        );
        self.mlp_silu_mul.compute(
            device,
            queue,
            &self.mlp_up_proj_buffer,
            &self.mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_down_proj_mul_mat.compute(
            device,
            queue,
            &self.mlp_up_proj_buffer,
            output_buffer,
            seq_len,
        );
    }
}
