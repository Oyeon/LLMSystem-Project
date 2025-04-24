// fused_tanh_marginloss_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

template <typename scalar_t>
__global__ void fused_tanh_marginloss_cuda_forward_kernel(
    const scalar_t* __restrict__ pos_input,
    const scalar_t* __restrict__ neg_input,
    scalar_t* __restrict__ pos_output,
    scalar_t* __restrict__ neg_output,
    scalar_t* __restrict__ loss_output,
    size_t n,
    float margin) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Compute tanh activations in-place
        pos_output[idx] = tanh(pos_input[idx]);
        neg_output[idx] = tanh(neg_input[idx]);
        
        // Compute loss
        scalar_t element_loss = max(scalar_t(0.0), margin - pos_output[idx] + neg_output[idx]);
        
        // Use atomicAdd for reduction, loss_output should be a single element tensor
        atomicAdd(loss_output, element_loss / n);
    }
}

template <typename scalar_t>
__global__ void fused_tanh_marginloss_cuda_backward_kernel(
    const scalar_t* __restrict__ grad_output,
    const scalar_t* __restrict__ pos_input,
    const scalar_t* __restrict__ neg_input,
    const scalar_t* __restrict__ pos_score,
    const scalar_t* __restrict__ neg_score,
    scalar_t* __restrict__ grad_pos_input,
    scalar_t* __restrict__ grad_neg_input,
    size_t n,
    float margin) {
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < n) {
        // Compute hinge loss gradient factor
        bool active = (margin - pos_score[idx] + neg_score[idx] > 0);
        scalar_t grad_factor = active ? (*grad_output / n) : 0;
        
        // Compute gradients with tanh backward: dtanh(x)/dx = 1 - tanh(x)^2
        scalar_t pos_tanh_grad = 1 - (pos_score[idx] * pos_score[idx]);
        scalar_t neg_tanh_grad = 1 - (neg_score[idx] * neg_score[idx]);
        
        // Accumulate gradients
        grad_pos_input[idx] = -grad_factor * pos_tanh_grad;  // Negative for pos_score
        grad_neg_input[idx] = grad_factor * neg_tanh_grad;   // Positive for neg_score
    }
}

torch::Tensor fused_tanh_marginloss_cuda_forward(
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const float margin) {
    
    auto pos_score = torch::empty_like(pos_input);
    auto neg_score = torch::empty_like(neg_input);
    auto loss = torch::zeros({1}, pos_input.options());
    
    const int threads = 1024;
    const int blocks = (pos_input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(pos_input.type(), "fused_tanh_marginloss_forward_cuda", ([&] {
        fused_tanh_marginloss_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            pos_input.data_ptr<scalar_t>(),
            neg_input.data_ptr<scalar_t>(),
            pos_score.data_ptr<scalar_t>(),
            neg_score.data_ptr<scalar_t>(),
            loss.data_ptr<scalar_t>(),
            pos_input.numel(),
            margin
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return torch::cat({loss, pos_score.reshape({-1, 1}), neg_score.reshape({-1, 1})}, 0);
}

std::vector<torch::Tensor> fused_tanh_marginloss_cuda_backward(
    const torch::Tensor& grad_output,
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const torch::Tensor& pos_score,
    const torch::Tensor& neg_score,
    const float margin) {
    
    auto grad_pos_input = torch::zeros_like(pos_input);
    auto grad_neg_input = torch::zeros_like(neg_input);
    
    const int threads = 1024;
    const int blocks = (pos_input.numel() + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(pos_input.type(), "fused_tanh_marginloss_backward_cuda", ([&] {
        fused_tanh_marginloss_cuda_backward_kernel<scalar_t><<<blocks, threads>>>(
            grad_output.data_ptr<scalar_t>(),
            pos_input.data_ptr<scalar_t>(),
            neg_input.data_ptr<scalar_t>(),
            pos_score.data_ptr<scalar_t>(),
            neg_score.data_ptr<scalar_t>(),
            grad_pos_input.data_ptr<scalar_t>(),
            grad_neg_input.data_ptr<scalar_t>(),
            pos_input.numel(),
            margin
        );
    }));
    
    // Check for errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
    }
    
    return {grad_pos_input, grad_neg_input};
}