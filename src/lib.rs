pub mod attention;
pub mod binary;
pub mod conv_silu;
pub mod delta_rule;
pub mod embedding_lookup;
pub mod gated_rms_norm;
pub mod layer_loop;
pub mod norm;
pub mod mamba_scan;
pub mod mlp;
pub mod mul_mat;
pub mod rope;
pub mod sigmoid_mul;
pub mod silu_mul;
pub mod slice_copy;

use tracing::debug;

pub fn log_tensor(name: &str, tensor: &safetensors::tensor::TensorView) {
    let num_elements: usize = tensor.shape().iter().product();
    if num_elements <= 64 {
        debug!("Tensor {}: {:?}", name, tensor);
    } else {
        debug!(
            "Tensor {}: dtype={:?}, shape={:?}",
            name,
            tensor.dtype(),
            tensor.shape()
        );
    }
}