import torch
import torch.nn as nn
from torch.autograd import Function
import fused_tanh_marginloss_cuda

class FusedTanhMarginlossFunction(Function):
    @staticmethod
    def forward(ctx, pos_input, neg_input, margin=1.0):
        # Save input tensors for backward pass
        ctx.save_for_backward(pos_input, neg_input)
        ctx.margin = margin
        
        # Store original tensor shapes for reshaping later
        ctx.pos_shape = pos_input.shape

        pos_input = pos_input.contiguous()
        neg_input = neg_input.contiguous()
        
        # Ensure inputs are flattened for the CUDA kernel
        pos_input_flat = pos_input.view(-1)
        neg_input_flat = neg_input.view(-1)
        
        # Call the CUDA forward function
        output = fused_tanh_marginloss_cuda.forward(pos_input_flat, neg_input_flat, margin)
        
        # First element is the loss
        loss = output[0].view(1)
        
        return loss
    
    @staticmethod
    def backward(ctx, grad_loss):
        pos_input, neg_input = ctx.saved_tensors
        margin = ctx.margin
        
        # Calculate tanh directly to use in backward
        pos_scores = torch.tanh(pos_input)
        neg_scores = torch.tanh(neg_input)
        
        # Flatten for CUDA kernel
        pos_input_flat = pos_input.reshape(-1)
        neg_input_flat = neg_input.reshape(-1)
        pos_scores_flat = pos_scores.reshape(-1)
        neg_scores_flat = neg_scores.reshape(-1)
        
        # Call the CUDA backward function
        grad_pos_input_flat, grad_neg_input_flat = fused_tanh_marginloss_cuda.backward(
            grad_loss, pos_input_flat, neg_input_flat, pos_scores_flat, neg_scores_flat, margin)
        
        # Reshape gradients back to original dimensions
        grad_pos_input = grad_pos_input_flat.reshape(ctx.pos_shape)
        grad_neg_input = grad_neg_input_flat.reshape(ctx.pos_shape)
        
        return grad_pos_input, grad_neg_input, None

class FusedTanhMarginloss(nn.Module):
    def __init__(self, margin=1.0, use_fused_kernel=True):
        super(FusedTanhMarginloss, self).__init__()
        self.margin = margin
        self.use_fused_kernel = use_fused_kernel
        self.standard_mr_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
    
    def forward(self, pos_input, neg_input):
        # Ensure inputs have the same shape
        if pos_input.shape != neg_input.shape:
            raise ValueError(f"Input shapes must match: got {pos_input.shape} and {neg_input.shape}")
            
        if self.use_fused_kernel and pos_input.is_cuda and neg_input.is_cuda:
            try:
                loss = FusedTanhMarginlossFunction.apply(
                    pos_input, neg_input, self.margin)
                return loss
            except Exception as e:
                print(f"Fused kernel failed, falling back to standard implementation: {e}")
                return self._standard_forward(pos_input, neg_input)
        else:
            # Fallback to standard implementation
            return self._standard_forward(pos_input, neg_input)
    
    def _standard_forward(self, pos_input, neg_input):
        pos_scores = torch.tanh(pos_input)
        neg_scores = torch.tanh(neg_input)
        loss = self.standard_mr_loss(pos_scores, neg_scores, torch.ones_like(pos_scores, device=pos_scores.device))
        return loss, pos_scores, neg_scores