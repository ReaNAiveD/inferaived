pub mod buffer_view;
pub mod dispatch;
pub mod embedding_lookup;
pub mod gpu_sampler;
pub mod kernels;
pub mod language_model;
pub mod layers;
pub mod lm_head;
pub mod sampling;

#[cfg(test)]
pub(crate) mod gpu_test_utils;

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
