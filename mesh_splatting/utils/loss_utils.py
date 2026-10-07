#
# The original code is under the following copyright:
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE_GS.md file.
#
# For inquiries contact george.drettakis@inria.fr
#
# The modifications of the code are under the following copyright:
# Copyright (C) 2024, University of Liege, KAUST and University of Oxford
# TELIM research group, http://www.telecom.ulg.ac.be/
# IVUL research group, https://ivul.kaust.edu.sa/
# VGG research group, https://www.robots.ox.ac.uk/~vgg/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp
import numpy as np

def vertex_depth_loss_hr(vertex_depth_out, image_2D, vertex_rendered, depth_full_hr, max_diff_threshold=0.5, iteration=10):
    assert depth_full_hr.dim() == 3 and depth_full_hr.size(0) == 1, "depth_full_hr must be (1,H,W)"

    v_depths = vertex_depth_out
    v_coords = image_2D.to(depth_full_hr)  # [y, x] as float

    H, W = depth_full_hr.shape[1], depth_full_hr.shape[2]

    # SWAPPED: v_coords stores [y, x] not [x, y]
    y = v_coords[:, 1]  # height dimension - FIRST COLUMN
    x = v_coords[:, 0]  # width dimension - SECOND COLUMN

    # inside image
    inside = (x >= 0) & (x < W) & (y >= 0) & (y < H)

    keep = inside
    
    if not torch.any(keep):
        print("NO values")
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)

    x = x[keep]; y = y[keep]; v_depths = v_depths[keep]

    # round to nearest pixel, then clamp for safety
    x_idx = torch.clamp(torch.round(x).long(), 0, W - 1)  # width indices
    y_idx = torch.clamp(torch.round(y).long(), 0, H - 1)  # height indices

    # Sampling with proper indices
    sampled = depth_full_hr[0, y_idx, x_idx]

    valid = torch.isfinite(sampled) & (sampled > 0)
    if not torch.any(valid):
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)

    diffs = torch.abs((v_depths[valid] - sampled[valid]))
    use = diffs < max_diff_threshold
    if not torch.any(use):
        return torch.tensor(0.0, device=vertex_depth_out.device, dtype=vertex_depth_out.dtype)
    
    return diffs[use].mean()


# def equilateral_regularizer(triangle_indices, vertices):

#     nan_mask = torch.isnan(triangle_indices).any(dim=(1, 2))
#     if nan_mask.any():
#         print("NaN detected in triangle(s):")
#         return torch.tensor(0.0, device=triangle_indices.device, dtype=triangle_indices.dtype)

#     v0 = vertices[triangle_indices[:, 1, :]] - vertices[triangle_indices[:, 0, :]]
#     v1 = vertices[triangle_indices[:, 2, :]] - vertices[triangle_indices[:, 0, :]]
#     cross = torch.cross(v0, v1, dim=1)
#     area = 0.5 * torch.norm(cross, dim=1)
#     return area.mean()


def equilateral_regularizer(triangle_indices, vertices, eps=1e-8):
    """
    Regularizes triangles based on deviation from 60-degree angles.
    
    Args:
        triangle_indices (torch.Tensor): (N, 3)
        vertices (torch.Tensor): (V, 3)
        eps (float): Small constant to avoid division by zero.
    """
    # 1. Gather vertices
    tri_verts = vertices[triangle_indices] # (N, 3, 3)
    v0, v1, v2 = tri_verts[:, 0], tri_verts[:, 1], tri_verts[:, 2]

    # 2. Define vectors for edges (normalized)
    # Edges meeting at v0
    e01 = F.normalize(v1 - v0, dim=-1, eps=eps)
    e02 = F.normalize(v2 - v0, dim=-1, eps=eps)
    
    # Edges meeting at v1
    e10 = -e01
    e12 = F.normalize(v2 - v1, dim=-1, eps=eps)
    
    # Edges meeting at v2
    e20 = -e02
    e21 = -e12

    # 3. Calculate cosines of the three internal angles
    # dot(A, B) = cos(theta) for normalized vectors
    cos_a = torch.sum(e01 * e02, dim=-1)
    cos_b = torch.sum(e10 * e12, dim=-1)
    cos_c = torch.sum(e20 * e21, dim=-1)

    # 4. Target is cos(60°) = 0.5
    loss_a = (cos_a - 0.5)**2
    loss_b = (cos_b - 0.5)**2
    loss_c = (cos_c - 0.5)**2

    # 5. Average over vertices and then triangles
    return (loss_a + loss_b + loss_c).mean() / 3.0


def u_shaped_opacity_loss(x, center=0.1, width=0.03):
    # Normalized distance to center (e.g., 0.1)
    penalty = torch.exp(-((x - center) ** 2) / (2 * width ** 2))
    return penalty.mean()

def binarization_loss(x, eps=1e-6):
    x = torch.clamp(x, eps, 1 - eps)  # avoid log(0)
    return -x * torch.log(x) - (1 - x) * torch.log(1 - x)


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def lp_loss(pred, target, p=0.7, eps=1e-6):
    """
    Computes Lp loss with 0 < p < 1.
    Args:
        pred: (N, C, H, W) predicted image
        target: (N, C, H, W) groundtruth image
        p: norm degree < 1
        eps: small constant for numerical stability
    """
    diff = torch.abs(pred - target) + eps
    loss = torch.pow(diff, p).mean()
    return loss

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


def point_to_normal(points1, points2=None):
    points = (
        points1[None] if points2 is None 
        else torch.stack([points1, points2],dim=0)
    )
    output = torch.zeros_like(points)
    dx = points[...,2:, 1:-1] - points[...,:-2, 1:-1]
    dy = points[...,1:-1, 2:] - points[...,1:-1, :-2]
    normal_map = torch.nn.functional.normalize(torch.cross(dx, dy, dim=1), dim=1)
    output[...,1:-1, 1:-1] = normal_map
    return (
        output[0] if points2 is None 
        else output
    )

def d2n_tblr(points: torch.Tensor, 
             k: int = 3, 
             d_min: float = 1e-3, 
             d_max: float = 10.0) -> torch.Tensor:
    """ points:     3D points in camera coordinates, shape: (B, 3, H, W)
        k:          neighborhood size
            e.g.)   If k=3, 3x3 neighborhood is used. Two vectors are defined by doing (top-bottom) and (left-right) 
                    Then the normals are computed via cross-product
        d_min/max:  Range of valid depth values 
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(points, (k,k,k,k), mode='constant', value=0)             # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:,2:,:,:] > d_min) & (points_pad[:,2:,:,:] < d_max) # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = points_pad[:, :, :H, k:k+W] - points_pad[:, :, 2*k:2*k+H, k:k+W]   # (B, 3, H, W)

    # horizontal vector (left - right)
    vec_hori = points_pad[:, :, k:k+H, :W] - points_pad[:, :, k:k+H, 2*k:2*k+W]   # (B, 3, H, W)

    # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
    valid_mask = valid_pad[:, :, k:k+H,     k:k+W       ] * \
                 valid_pad[:, :, :H,        k:k+W       ] * \
                 valid_pad[:, :, 2*k:2*k+H, k:k+W       ] * \
                 valid_pad[:, :, k:k+H,     :W          ] * \
                 valid_pad[:, :, k:k+H,     2*k:2*k+W   ]
    valid_mask = valid_mask > 0.5
    
    # get cross product (B, 3, H, W)
    cross_product = - torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)
   
    return normal, valid_mask