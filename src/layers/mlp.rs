use safetensors::{SafeTensors, tensor::TensorView};

use crate::{
    buffer_view::BufferView,
    kernels::{
        mul_mat::{MulMatWebgpu, MulMatWebgpuRunner},
        silu_mul::{SiluMulInplaceWebgpu, SiluMulInplaceWebgpuRunner},
    },
    log_tensor,
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

    pub fn from_weights<'data>(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        gate_proj_weight: TensorView<'data>,
        up_proj_weight: TensorView<'data>,
        down_proj_weight: TensorView<'data>,
        hidden_size: usize,
        intermediate_size: usize,
    ) -> Self {
        debug_assert_eq!(
            gate_proj_weight.shape()[0],
            intermediate_size,
            "mlp gate_proj height does not match intermediate_size",
        );
        debug_assert_eq!(
            gate_proj_weight.shape()[1],
            hidden_size,
            "mlp gate_proj width does not match hidden_size",
        );
        let mlp_gate_proj_mul_mat = MulMatWebgpu::new(device, queue, gate_proj_weight);
        debug_assert_eq!(
            up_proj_weight.shape()[0],
            intermediate_size,
            "mlp up_proj height does not match intermediate_size",
        );
        debug_assert_eq!(
            up_proj_weight.shape()[1],
            hidden_size,
            "mlp up_proj width does not match hidden_size",
        );
        let mlp_up_proj_mul_mat = MulMatWebgpu::new(device, queue, up_proj_weight);
        let mlp_silu_mul = SiluMulInplaceWebgpu::new(device, intermediate_size);
        debug_assert_eq!(
            down_proj_weight.shape()[0],
            hidden_size,
            "mlp down_proj height does not match hidden_size",
        );
        debug_assert_eq!(
            down_proj_weight.shape()[1],
            intermediate_size,
            "mlp down_proj width does not match intermediate_size",
        );
        let mlp_down_proj_mul_mat = MulMatWebgpu::new(device, queue, down_proj_weight);
        Self {
            intermediate_size,
            mlp_gate_proj_mul_mat,
            mlp_up_proj_mul_mat,
            mlp_silu_mul,
            mlp_down_proj_mul_mat,
        }
    }

    pub fn plan(
        &self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        input: BufferView<'_>,
        output: BufferView<'_>,
    ) -> MlpRunners {
        debug_assert_eq!(
            input.shape[0], output.shape[0],
            "mlp: outer dim mismatch (input={}, output={})",
            input.shape[0], output.shape[0],
        );
        let num_rows = input.shape[0];
        let elem_size = std::mem::size_of::<f32>() as u32;
        let intermediate_size = self.intermediate_size as u32;
        let intermediate_total_bytes =
            (intermediate_size as u64) * (elem_size as u64) * (num_rows as u64);
        let gate_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mlp_runners/gate_proj_buffer"),
            size: intermediate_total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let gate_view =
            BufferView::new_2d_tight(&gate_buffer, num_rows, intermediate_size, elem_size);
        let up_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mlp_runners/up_proj_buffer"),
            size: intermediate_total_bytes,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let up_view = BufferView::new_2d_tight(&up_buffer, num_rows, intermediate_size, elem_size);
        let gate_runner = self
            .mlp_gate_proj_mul_mat
            .plan(device, queue, input, gate_view);
        let up_runner = self.mlp_up_proj_mul_mat.plan(device, queue, input, up_view);
        let silu_mul_runner = self.mlp_silu_mul.plan(device, queue, up_view, gate_view);
        let down_runner = self
            .mlp_down_proj_mul_mat
            .plan(device, queue, up_view, output);
        MlpRunners {
            gate_runner,
            up_runner,
            silu_mul_runner,
            down_runner,
        }
    }
}

/// Cached runners for one MLP forward pass. Buffer references live
/// inside the wgpu bind groups; nothing extra to keep alive here.
pub struct MlpRunners {
    gate_runner: MulMatWebgpuRunner,
    up_runner: MulMatWebgpuRunner,
    silu_mul_runner: SiluMulInplaceWebgpuRunner,
    down_runner: MulMatWebgpuRunner,
}

impl MlpRunners {
    pub fn forward(&self, cpass: &mut wgpu::ComputePass<'_>) {
        self.gate_runner.forward(cpass);
        self.up_runner.forward(cpass);
        self.silu_mul_runner.forward(cpass);
        self.down_runner.forward(cpass);
    }
}
