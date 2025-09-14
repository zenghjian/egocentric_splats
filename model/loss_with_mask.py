# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
# Loss functions with mask support (inspired by EgoLifter implementation)

import torch
import torch.nn.functional as F

def l1_loss(pred, gt, mask=None):
    """
    L1 loss with optional mask support (EgoLifter style).
    
    Args:
        pred: Predicted values
        gt: Ground truth values  
        mask: Binary mask (1 for valid pixels, 0 for invalid)
    
    Returns:
        Masked L1 loss
    """
    if mask is None:
        return torch.abs((pred - gt)).mean()
    else:
        # Ensure mask is on same device and expand to match shape
        mask = mask.to(pred.device)
        if mask.dim() < pred.dim():
            mask = mask.expand(pred.shape)
        
        # Calculate loss only on masked regions
        loss_total = (pred * mask - gt * mask).abs().sum()
        loss_count = mask.sum()
        
        # Avoid division by zero
        if loss_count == 0:
            return torch.tensor(0.0, device=pred.device)
        
        return loss_total / loss_count

def l2_loss(pred, gt, mask=None):
    """
    L2 loss with optional mask support (EgoLifter style).
    """
    if mask is None:
        return ((pred - gt) ** 2).mean()
    else:
        mask = mask.to(pred.device)
        if mask.dim() < pred.dim():
            mask = mask.expand(pred.shape)
        
        loss_total = ((pred * mask - gt * mask) ** 2).sum()
        loss_count = mask.sum()
        
        if loss_count == 0:
            return torch.tensor(0.0, device=pred.device)
            
        return loss_total / loss_count

def psnr(pred, gt, mask=None):
    """
    PSNR with optional mask support.
    """
    mse = l2_loss(pred, gt, mask=mask)
    if mse == 0:
        return torch.tensor(100.0, device=pred.device)  # Perfect match
    return 20 * torch.log10(1.0 / torch.sqrt(mse))

def compute_recon_metrics(image, gt_image, mask=None):
    """
    Compute reconstruction metrics with optional mask (EgoLifter style).
    
    Args:
        image: Predicted image
        gt_image: Ground truth image
        mask: Optional mask for filtering regions
    
    Returns:
        Dictionary of metrics
    """
    L_l1 = l1_loss(image, gt_image, mask=mask).mean().double()
    L_l2 = l2_loss(image, gt_image, mask=mask).mean().double()
    metric_psnr = psnr(image, gt_image, mask=mask).mean().double()
    
    metrics = {
        "L_l1": L_l1,
        "L_l2": L_l2,
        "psnr": metric_psnr,
    }
    
    return metrics