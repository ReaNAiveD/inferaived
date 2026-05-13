use safetensors::SafeTensors;

use crate::{log_tensor, mul_mat::MulMatWebgpu, silu_mul::SiluMulInplaceWebgpu};

pub struct MultiLayerPerceptron {
    intermediate_size: usize,
    mlp_gate_proj_mul_mat: MulMatWebgpu,
    mlp_up_proj_mul_mat: MulMatWebgpu,
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
        intermediate_size: usize,
    ) -> Self {
        let mlp_gate_proj_weight_name = format!("{}.mlp.gate_proj.weight", weight_prefix);
        let mlp_gate_proj_weight = tensor.tensor(&mlp_gate_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_gate_proj_weight_name
        ));
        log_tensor(&mlp_gate_proj_weight_name, &mlp_gate_proj_weight);
        debug_assert_eq!(
            mlp_gate_proj_weight.shape()[0] as usize,
            intermediate_size,
            "{} height does not match intermediate_size",
            mlp_gate_proj_weight_name
        );
        let mlp_gate_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_gate_proj_weight, hidden_size);
        let mlp_up_proj_weight_name = format!("{}.mlp.up_proj.weight", weight_prefix);
        let mlp_up_proj_weight = tensor.tensor(&mlp_up_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_up_proj_weight_name
        ));
        log_tensor(&mlp_up_proj_weight_name, &mlp_up_proj_weight);
        debug_assert_eq!(
            mlp_up_proj_weight.shape()[0] as usize,
            intermediate_size,
            "{} height does not match intermediate_size",
            mlp_up_proj_weight_name
        );
        let mlp_up_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_up_proj_weight, hidden_size);
        let mlp_silu_mul = SiluMulInplaceWebgpu::new(&device, intermediate_size);
        let mlp_down_proj_weight_name = format!("{}.mlp.down_proj.weight", weight_prefix);
        let mlp_down_proj_weight = tensor.tensor(&mlp_down_proj_weight_name).expect(&format!(
            "Failed to get tensor for {}",
            mlp_down_proj_weight_name
        ));
        log_tensor(&mlp_down_proj_weight_name, &mlp_down_proj_weight);
        debug_assert_eq!(
            mlp_down_proj_weight.shape()[0] as usize,
            hidden_size,
            "{} height does not match hidden_size",
            mlp_down_proj_weight_name
        );
        let mlp_down_proj_mul_mat =
            MulMatWebgpu::new(&device, &queue, mlp_down_proj_weight, intermediate_size);
        Self {
            intermediate_size,
            mlp_gate_proj_mul_mat,
            mlp_up_proj_mul_mat,
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
        let intermediate_buffer_size =
            (seq_len * self.intermediate_size * std::mem::size_of::<f32>()) as wgpu::BufferAddress;
        let mlp_gate_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mlp/gate_proj_buffer"),
            size: intermediate_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let mlp_up_proj_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mlp/up_proj_buffer"),
            size: intermediate_buffer_size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        self.mlp_gate_proj_mul_mat.compute(
            device,
            queue,
            input_buffer,
            &mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_up_proj_mul_mat.compute(
            device,
            queue,
            input_buffer,
            &mlp_up_proj_buffer,
            seq_len,
        );
        self.mlp_silu_mul.compute(
            device,
            queue,
            &mlp_up_proj_buffer,
            &mlp_gate_proj_buffer,
            seq_len,
        );
        self.mlp_down_proj_mul_mat.compute(
            device,
            queue,
            &mlp_up_proj_buffer,
            output_buffer,
            seq_len,
        );
    }
}
