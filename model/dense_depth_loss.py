# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List


def calculate_dense_depth_loss(
    render_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    loss_type: str = "l1",
    use_inverse_depth: bool = True,
    robust_loss_delta: float = 0.5,
    depth_scale_factor: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Calculate dense depth supervision loss.
    
    Args:
        render_depth: Rendered depth map from Gaussian Splatting (B, H, W) or (H, W)
        gt_depth: Ground truth depth map from ADT (B, H, W) or (H, W)
        valid_mask: Boolean mask of valid depth pixels (B, H, W) or (H, W)
        loss_type: Type of loss - "l1", "l2", "huber", or "combined"
        use_inverse_depth: Whether to compute loss in inverse depth space
        robust_loss_delta: Delta parameter for Huber loss
        depth_scale_factor: Scale factor to align depth units if needed
        
    Returns:
        Dictionary containing computed losses
    """
    # Ensure tensors have batch dimension
    if render_depth.dim() == 2:
        render_depth = render_depth.unsqueeze(0)
    if gt_depth.dim() == 2:
        gt_depth = gt_depth.unsqueeze(0)
    if valid_mask is not None and valid_mask.dim() == 2:
        valid_mask = valid_mask.unsqueeze(0)
    
    # Apply depth scale factor if needed
    render_depth = render_depth * depth_scale_factor
    
    # Create valid mask if not provided
    if valid_mask is None:
        valid_mask = (gt_depth > 0.1) & (gt_depth < 50.0) & (render_depth > 0.1)
    else:
        # Combine with depth validity checks
        valid_mask = valid_mask & (gt_depth > 0.1) & (gt_depth < 50.0) & (render_depth > 0.1)
    
    # Get number of valid pixels for normalization
    num_valid = valid_mask.sum() + 1e-6  # Add small epsilon to avoid division by zero
    
    losses = {}
    
    if use_inverse_depth:
        # Convert to inverse depth space (more robust for large depth ranges)
        render_inv_depth = torch.where(
            render_depth > 0.1,
            1.0 / render_depth,
            torch.zeros_like(render_depth)
        )
        gt_inv_depth = torch.where(
            gt_depth > 0.1,
            1.0 / gt_depth,
            torch.zeros_like(gt_depth)
        )
        
        # Compute difference in inverse depth space
        diff = render_inv_depth - gt_inv_depth
    else:
        # Compute difference in regular depth space
        diff = render_depth - gt_depth
    
    # Apply mask
    diff_masked = diff * valid_mask.float()
    
    # Compute different loss types
    if loss_type == "l1" or loss_type == "combined":
        l1_loss = torch.abs(diff_masked).sum() / num_valid
        losses["l1"] = l1_loss
    
    if loss_type == "l2" or loss_type == "combined":
        l2_loss = (diff_masked ** 2).sum() / num_valid
        losses["l2"] = l2_loss
    
    if loss_type == "huber" or loss_type == "combined":
        # Huber loss (smooth L1) - more robust to outliers
        huber_loss = F.huber_loss(
            render_inv_depth if use_inverse_depth else render_depth,
            gt_inv_depth if use_inverse_depth else gt_depth,
            reduction='none',
            delta=robust_loss_delta
        )
        huber_loss_masked = huber_loss * valid_mask.float()
        losses["huber"] = huber_loss_masked.sum() / num_valid
    
    # Add relative depth loss (scale-invariant)
    if "combined" in loss_type:
        # Log depth difference (scale-invariant loss)
        eps = 1e-6
        log_diff = torch.log(render_depth + eps) - torch.log(gt_depth + eps)
        log_diff_masked = log_diff * valid_mask.float()
        
        # Scale-invariant loss from Eigen et al.
        scale_invariant_loss = (log_diff_masked ** 2).sum() / num_valid
        scale_invariant_loss -= 0.5 * (log_diff_masked.sum() / num_valid) ** 2
        losses["scale_invariant"] = scale_invariant_loss
        
        # Gradient matching loss for sharp depth discontinuities
        if render_depth.shape[-1] > 1 and render_depth.shape[-2] > 1:
            grad_loss = compute_depth_gradient_loss(
                render_depth, gt_depth, valid_mask
            )
            losses["gradient"] = grad_loss
    
    # Compute metrics for logging
    with torch.no_grad():
        # Absolute relative error
        rel_error = (torch.abs(diff_masked) / (gt_depth + 1e-6))
        losses["abs_rel_error"] = (rel_error * valid_mask.float()).sum() / num_valid
        
        # RMSE
        rmse = torch.sqrt((diff_masked ** 2).sum() / num_valid)
        losses["rmse"] = rmse
        
        # Threshold accuracy (δ < 1.25)
        ratio = torch.maximum(
            render_depth / (gt_depth + 1e-6),
            gt_depth / (render_depth + 1e-6)
        )
        threshold_1_25 = ((ratio < 1.25) & valid_mask).float().sum() / num_valid
        losses["threshold_1.25"] = threshold_1_25
    
    return losses


def compute_depth_gradient_loss(
    render_depth: torch.Tensor,
    gt_depth: torch.Tensor,
    valid_mask: torch.Tensor,
) -> torch.Tensor:
    """
    Compute gradient matching loss for depth maps.
    
    Args:
        render_depth: Rendered depth map (B, H, W)
        gt_depth: Ground truth depth map (B, H, W)
        valid_mask: Valid pixel mask (B, H, W)
        
    Returns:
        Gradient matching loss
    """
    # Sobel filters for gradient computation
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], 
                           dtype=torch.float32, device=render_depth.device)
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], 
                           dtype=torch.float32, device=render_depth.device)
    
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    
    # Add channel dimension for convolution
    render_depth_expanded = render_depth.unsqueeze(1) if render_depth.dim() == 3 else render_depth
    gt_depth_expanded = gt_depth.unsqueeze(1) if gt_depth.dim() == 3 else gt_depth
    valid_mask_expanded = valid_mask.unsqueeze(1) if valid_mask.dim() == 3 else valid_mask
    
    # Compute gradients
    render_grad_x = F.conv2d(render_depth_expanded, sobel_x, padding=1)
    render_grad_y = F.conv2d(render_depth_expanded, sobel_y, padding=1)
    gt_grad_x = F.conv2d(gt_depth_expanded, sobel_x, padding=1)
    gt_grad_y = F.conv2d(gt_depth_expanded, sobel_y, padding=1)
    
    # Compute gradient magnitude
    render_grad_mag = torch.sqrt(render_grad_x ** 2 + render_grad_y ** 2 + 1e-6)
    gt_grad_mag = torch.sqrt(gt_grad_x ** 2 + gt_grad_y ** 2 + 1e-6)
    
    # Compute loss
    grad_diff = torch.abs(render_grad_mag - gt_grad_mag)
    
    # Erode mask to avoid edge artifacts
    kernel = torch.ones(1, 1, 3, 3, device=valid_mask.device)
    valid_mask_eroded = F.conv2d(valid_mask_expanded.float(), kernel, padding=1) == 9
    
    # Apply mask and normalize
    grad_loss = (grad_diff * valid_mask_eroded.float()).sum()
    grad_loss = grad_loss / (valid_mask_eroded.sum() + 1e-6)
    
    return grad_loss.squeeze()


def combine_depth_losses(
    sparse_loss: Optional[Dict[str, torch.Tensor]] = None,
    dense_loss: Optional[Dict[str, torch.Tensor]] = None,
    sparse_weight: float = 0.1,
    dense_weight: float = 1.0,
) -> Dict[str, torch.Tensor]:
    """
    Combine sparse and dense depth losses.
    
    Args:
        sparse_loss: Loss from sparse depth supervision
        dense_loss: Loss from dense depth supervision
        sparse_weight: Weight for sparse loss
        dense_weight: Weight for dense loss
        
    Returns:
        Combined loss dictionary
    """
    combined_losses = {}
    
    if sparse_loss is not None:
        for key, value in sparse_loss.items():
            combined_losses[f"sparse_{key}"] = value * sparse_weight
    
    if dense_loss is not None:
        for key, value in dense_loss.items():
            combined_losses[f"dense_{key}"] = value * dense_weight
    
    # Compute total depth loss
    total_loss = torch.tensor(0.0, device=next(iter(combined_losses.values())).device)
    
    # Only add main loss components to total (not metrics)
    loss_keys = ["sparse_huber", "sparse_l1", "dense_huber", "dense_l1", "dense_scale_invariant"]
    for key in loss_keys:
        if key in combined_losses:
            total_loss += combined_losses[key]
    
    combined_losses["total_depth_loss"] = total_loss
    
    return combined_losses