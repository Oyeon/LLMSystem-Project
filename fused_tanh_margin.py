import torch
import torch.nn as nn
from torch.autograd import Function
import fused_tanh_marginloss_cuda

class FusedTanhMarginlossFunction(Function):
    @staticmethod
    def forward(ctx, pos_input, neg_input, margin=1.0):
        output = fused_tanh_marginloss_cuda.forward(pos_input, neg_input, margin)
        
        # The output contains [loss, pos_scores, neg_scores]
        loss = output[0].view(1)
        pos_scores = output[1:1+pos_input.numel()].view_as(pos_input)
        neg_scores = output[1+pos_input.numel():].view_as(neg_input)
        
        # Save for backward
        ctx.save_for_backward(pos_input, neg_input, pos_scores, neg_scores)
        ctx.margin = margin
        
        return loss, pos_scores, neg_scores
    
    @staticmethod
    def backward(ctx, grad_loss, grad_pos_scores, grad_neg_scores):
        pos_input, neg_input, pos_scores, neg_scores = ctx.saved_tensors
        margin = ctx.margin
        
        grad_pos_input, grad_neg_input = fused_tanh_marginloss_cuda.backward(
            grad_loss, pos_input, neg_input, pos_scores, neg_scores, margin)
        
        # Add gradients from tanh outputs if they're used elsewhere
        if grad_pos_scores is not None:
            tanh_grad_pos = (1 - pos_scores * pos_scores) * grad_pos_scores
            grad_pos_input = grad_pos_input + tanh_grad_pos
            
        if grad_neg_scores is not None:
            tanh_grad_neg = (1 - neg_scores * neg_scores) * grad_neg_scores
            grad_neg_input = grad_neg_input + tanh_grad_neg
        
        return grad_pos_input, grad_neg_input, None

class FusedTanhMarginloss(nn.Module):
    def __init__(self, margin=1.0, use_fused_kernel=True):
        super(FusedTanhMarginloss, self).__init__()
        self.margin = margin
        self.use_fused_kernel = use_fused_kernel
        self.standard_mr_loss = nn.MarginRankingLoss(margin=margin, reduction='mean')
    
    def forward(self, pos_input, neg_input):
        if self.use_fused_kernel and pos_input.is_cuda and neg_input.is_cuda:
            loss, pos_scores, neg_scores = FusedTanhMarginlossFunction.apply(
                pos_input, neg_input, self.margin)
            return loss, pos_scores, neg_scores
        else:
            # Fallback to standard implementation
            pos_scores = torch.tanh(pos_input)
            neg_scores = torch.tanh(neg_input)
            loss = self.standard_mr_loss(pos_scores, neg_scores, torch.ones_like(pos_scores))
            return loss, pos_scores, neg_scores