// fused_tanh_marginloss_cuda_kernel.cu
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>

__device__ inline float tanh_fast(float x) {
    float exp2x = expf(2.0f * x);
    return (exp2x - 1.0f) / (exp2x + 1.0f);
}

template <typename scalar_t>
__global__ void fused_tanh_marginloss_cuda_forward_kernel(
    const scalar_t* __restrict__ pos_input,
    const scalar_t* __restrict__ neg_input,
    scalar_t* __restrict__ pos_output,
    scalar_t* __restrict__ neg_output,
    scalar_t* __restrict__ loss_output,
    size_t n,
    float margin) {
    
    // Shared memory for reduction
    __shared__ scalar_t shared_loss[1024];
    
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int tid = threadIdx.x;
    
    // Initialize local loss
    scalar_t local_loss = 0;
    
    if (idx < n) {
        // Compute tanh activations in-place
        pos_output[idx] = tanh_fast(pos_input[idx]);
        neg_output[idx] = tanh_fast(neg_input[idx]);
        
        // Compute loss
        local_loss = max(scalar_t(0.0), margin - pos_output[idx] + neg_output[idx]);
        local_loss /= n;
    }
    
    // Store in shared memory
    shared_loss[tid] = local_loss;
    __syncthreads();
    
    // Parallel reduction in shared memory
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            shared_loss[tid] += shared_loss[tid + stride];
        }
        __syncthreads();
    }
    
    // Write result to global memory
    if (tid == 0) {
        atomicAdd(loss_output, shared_loss[0]);
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
        // Compute margin comparison
        scalar_t margin_diff = margin - pos_score[idx] + neg_score[idx];
        
        // Compute tanh gradients
        scalar_t pos_tanh_grad = 1 - (pos_score[idx] * pos_score[idx]);
        scalar_t neg_tanh_grad = 1 - (neg_score[idx] * neg_score[idx]);
        
        // Use multiplication instead of branching to reduce divergence
        scalar_t grad_factor = (margin_diff > 0) ? (*grad_output / n) : 0;
        
        // Accumulate gradients
        grad_pos_input[idx] = -grad_factor * pos_tanh_grad;
        grad_neg_input[idx] = grad_factor * neg_tanh_grad;
    }
}

torch::Tensor fused_tanh_marginloss_cuda_forward(
    const torch::Tensor& pos_input,
    const torch::Tensor& neg_input,
    const float margin) {
    
    auto options = pos_input.options();
    auto numel = pos_input.numel();
    
    // Allocate a single contiguous tensor for all outputs
    auto output = torch::empty({1 + 2 * numel}, options);

    auto pos_score = torch::empty_like(pos_input);
    auto neg_score = torch::empty_like(neg_input);
    auto loss = torch::zeros({1}, pos_input.options());
    
    // Initialize loss to zero
    output[0] = 0;
    
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;
    
    AT_DISPATCH_FLOATING_TYPES(pos_input.scalar_type(), "fused_tanh_marginloss_forward_cuda", ([&] {
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
    
    // Store original sizes to use later
    auto sizes = pos_input.sizes().vec();
    auto flattened_size = pos_input.numel();
    
    // For the return tensor, concatenate loss, pos_score flattened, and neg_score flattened
    auto concat_size = 1 + 2 * flattened_size;
    
    // Copy loss to first element
    output[0] = loss[0];
    
    return output;
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
    
    AT_DISPATCH_FLOATING_TYPES(pos_input.scalar_type(), "fused_tanh_marginloss_backward_cuda", ([&] {
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