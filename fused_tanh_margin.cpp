// fused_tanh_marginloss_cuda.cpp
#include <torch/extension.h>

// Forward declarations
torch::Tensor fused_tanh_marginloss_cuda_forward(
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const float margin);

std::vector<torch::Tensor> fused_tanh_marginloss_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const torch::Tensor& pos_score,
    const torch::Tensor& neg_score,
    const float margin);

// C++ interface
torch::Tensor fused_tanh_marginloss_forward(
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const float margin) {
    
    // Input validation
    TORCH_CHECK(pos_input.device().is_cuda(), "pos_input must be a CUDA tensor");
    TORCH_CHECK(neg_input.device().is_cuda(), "neg_input must be a CUDA tensor");
    TORCH_CHECK(pos_input.sizes() == neg_input.sizes(), "pos_input and neg_input must have the same shape");
    
    return fused_tanh_marginloss_cuda_forward(pos_input, neg_input, margin);
}

std::vector<torch::Tensor> fused_tanh_marginloss_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const torch::Tensor& pos_score,
    const torch::Tensor& neg_score,
    const float margin) {
    
    return fused_tanh_marginloss_cuda_backward(
        grad_output, pos_input, neg_input, pos_score, neg_score, margin);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &fused_tanh_marginloss_forward, "Fused TanH MarginRankingLoss forward");
    m.def("backward", &fused_tanh_marginloss_backward, "Fused TanH MarginRankingLoss backward");
}