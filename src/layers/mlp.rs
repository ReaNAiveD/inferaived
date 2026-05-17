use safetensors::SafeTensors;

use crate::{
    buffer_view::BufferView,
    kernels::{mul_mat::MulMatWebgpu, silu_mul::SiluMulInplaceWebgpu},
    log_tensor,
    scratch_pool::{ScratchPool, ScratchSlot},
};

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
        debug_assert_eq!(
            mlp_gate_proj_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            mlp_gate_proj_weight_name
        );
        let mlp_gate_proj_mul_mat = MulMatWebgpu::new(&device, &queue, mlp_gate_proj_weight);
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
        debug_assert_eq!(
            mlp_up_proj_weight.shape()[1] as usize,
            hidden_size,
            "{} width does not match hidden_size",
            mlp_up_proj_weight_name
        );
        let mlp_up_proj_mul_mat = MulMatWebgpu::new(&device, &queue, mlp_up_proj_weight);
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
        debug_assert_eq!(
            mlp_down_proj_weight.shape()[1] as usize,
            intermediate_size,
            "{} width does not match intermediate_size",
            mlp_down_proj_weight_name
        );
        let mlp_down_proj_mul_mat = MulMatWebgpu::new(&device, &queue, mlp_down_proj_weight);
        Self {
            intermediate_size,
            mlp_gate_proj_mul_mat,
            mlp_up_proj_mul_mat,
            mlp_silu_mul,
            mlp_down_proj_mul_mat,
        }
    }

    /// Run the SwiGLU MLP block over `input.row_count` token rows.
    pub fn forward(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        scratch: &ScratchPool,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) {
        debug_assert_eq!(
            input.shape[0], output.shape[0],
            "mlp: outer dim mismatch (input={}, output={})",
            input.shape[0], output.shape[0],
        );
        let num_rows = input.shape[0];
        debug_assert_eq!(
            scratch.feature_dim(ScratchSlot::MlpGate),
            self.intermediate_size as u32,
            "mlp: ScratchPool MlpGate dim does not match layer intermediate_size",
        );
        let gate_view = scratch.view_2d(ScratchSlot::MlpGate, num_rows);
        let up_view = scratch.view_2d(ScratchSlot::MlpUp, num_rows);
        self.mlp_gate_proj_mul_mat
            .forward(device, queue, input, gate_view);
        self.mlp_up_proj_mul_mat
            .forward(device, queue, input, up_view);
        self.mlp_silu_mul.forward(device, queue, up_view, gate_view);
        self.mlp_down_proj_mul_mat
            .forward(device, queue, up_view, output);
    }
}
