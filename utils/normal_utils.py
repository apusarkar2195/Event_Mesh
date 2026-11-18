import numpy as np

import torch
import torch.nn.functional as F

import torch.nn as nn


def intrins_to_intrins_inv(intrins):
    """intrins to intrins_inv

    NOTE: top-left is (0,0)
    """
    if torch.is_tensor(intrins):
        intrins_inv = torch.zeros_like(intrins)
    elif type(intrins) is np.ndarray:
        intrins_inv = np.zeros_like(intrins)
    else:
        raise Exception("intrins should be torch tensor or numpy array")

    intrins_inv[0, 0] = 1 / intrins[0, 0]
    intrins_inv[0, 2] = -intrins[0, 2] / intrins[0, 0]
    intrins_inv[1, 1] = 1 / intrins[1, 1]
    intrins_inv[1, 2] = -intrins[1, 2] / intrins[1, 1]
    intrins_inv[2, 2] = 1.0
    return intrins_inv


def get_cam_coords(intrins_inv, depth):
    """camera coordinates from intrins_inv and depth

    NOTE: intrins_inv should be a torch tensor of shape (B, 3, 3)
    NOTE: depth should be a torch tensor of shape (B, 1, H, W)
    NOTE: top-left is (0,0)
    """
    assert torch.is_tensor(intrins_inv) and intrins_inv.ndim == 3
    assert torch.is_tensor(depth) and depth.ndim == 4
    assert intrins_inv.dtype == depth.dtype
    assert intrins_inv.device == depth.device
    B, _, H, W = depth.size()

    u_range = (
        torch.arange(W, dtype=depth.dtype, device=depth.device).view(1, W).expand(H, W)
    )  # (H, W)
    v_range = (
        torch.arange(H, dtype=depth.dtype, device=depth.device).view(H, 1).expand(H, W)
    )  # (H, W)
    ones = torch.ones(H, W, dtype=depth.dtype, device=depth.device)
    pixel_coords = (
        torch.stack((u_range, v_range, ones), dim=0).unsqueeze(0).repeat(B, 1, 1, 1)
    )  # (B, 3, H, W)
    pixel_coords = pixel_coords.view(B, 3, H * W)  # (B, 3, H*W)

    cam_coords = intrins_inv.bmm(pixel_coords).view(B, 3, H, W)
    cam_coords = cam_coords * depth
    return cam_coords


def tensor_to_numpy(tensor_in):
    """torch tensor to numpy array"""
    if tensor_in is not None:
        if tensor_in.ndim == 3:
            # (C, H, W) -> (H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(1, 2, 0).numpy()
        elif tensor_in.ndim == 4:
            # (B, C, H, W) -> (B, H, W, C)
            tensor_in = tensor_in.detach().cpu().permute(0, 2, 3, 1).numpy()
        else:
            raise Exception("invalid tensor size")
    return tensor_in


def normal_to_rgb(normal, normal_mask=None):
    """surface normal map to RGB
    (used for visualization)

    NOTE: x, y, z are mapped to R, G, B
    NOTE: [-1, 1] are mapped to [0, 255]
    """
    if torch.is_tensor(normal):
        normal = tensor_to_numpy(normal)
        normal_mask = tensor_to_numpy(normal_mask)

    normal_norm = np.linalg.norm(normal, axis=-1, keepdims=True)
    normal_norm[normal_norm < 1e-12] = 1e-12
    normal = normal / normal_norm

    normal_rgb = (((normal + 1) * 0.5) * 255).astype(np.uint8)
    if normal_mask is not None:
        normal_rgb = normal_rgb * normal_mask  # (B, H, W, 3)
    return normal_rgb


def d2n_tblr(
    points: torch.Tensor, k: int = 3, d_min: float = 1e-3, d_max: float = 10.0
) -> torch.Tensor:
    """points:     3D points in camera coordinates, shape: (B, 3, H, W)
    k:          neighborhood size
        e.g.)   If k=3, 3x3 neighborhood is used. Two vectors are defined by doing (top-bottom) and (left-right)
                Then the normals are computed via cross-product
    d_min/max:  Range of valid depth values
    """
    k = (k - 1) // 2

    B, _, H, W = points.size()
    points_pad = F.pad(
        points, (k, k, k, k), mode="constant", value=0
    )  # (B, 3, k+H+k, k+W+k)
    valid_pad = (points_pad[:, 2:, :, :] > d_min) & (
        points_pad[:, 2:, :, :] < d_max
    )  # (B, 1, k+H+k, k+W+k)
    valid_pad = valid_pad.float()

    # vertical vector (top - bottom)
    vec_vert = (
        points_pad[:, :, :H, k : k + W] - points_pad[:, :, 2 * k : 2 * k + H, k : k + W]
    )  # (B, 3, H, W)

    # horizontal vector (left - right)
    vec_hori = (
        points_pad[:, :, k : k + H, :W] - points_pad[:, :, k : k + H, 2 * k : 2 * k + W]
    )  # (B, 3, H, W)

    # valid_mask (all five depth values - center/top/bottom/left/right should be valid)
    valid_mask = (
        valid_pad[:, :, k : k + H, k : k + W]
        * valid_pad[:, :, :H, k : k + W]
        * valid_pad[:, :, 2 * k : 2 * k + H, k : k + W]
        * valid_pad[:, :, k : k + H, :W]
        * valid_pad[:, :, k : k + H, 2 * k : 2 * k + W]
    )
    valid_mask = valid_mask > 0.5

    # get cross product (B, 3, H, W)
    cross_product = -torch.linalg.cross(vec_vert, vec_hori, dim=1)
    normal = F.normalize(cross_product, p=2.0, dim=1, eps=1e-12)

    return normal, valid_mask





class Depth2normal(nn.Module):
    def __init__(self, 
                 d_min: float = 0.0, 
                 d_max: float = 10.0, 
                 k: int = 3, 
                 d: int = 1, 
                 min_nghbr: int = 4, 
                 gamma: float = None, 
                 gamma_exception: bool = False):
        super(Depth2normal, self).__init__()

        # range of valid depth values
        # if the depth is outside this range, it will be considered invalid
        self.d_min = d_min
        self.d_max = d_max

        # neighborhood size, k x k neighborhood around each pixel will be considered
        self.k = k

        # spacing between the nghbrs (dilation)
        self.d = d

        # if the difference between the center depth and nghbr depth is larger than this, it will be ignored
        # e.g. gamma=0.05 means that a nghbr pixel is ignored if its depth is more than 5% different from the center pixel
        self.gamma = gamma  

        # minimum number of nghbr pixels
        # if the number of valid nghbr pixels is below this value, the normals would be considered invalid
        self.min_nghbr = min_nghbr

        # if the normal of a flat surface is near-vertical to the viewing direction, the depth gradient will be very high,
        # and most nghbr pixels would not pass the above "gamma" test
        # this can be a problem when using datasets like Virtual KITTI (i.e. the ones with large horizontal surfaces)
        # if gamma_exception is set to True, 
        # the "gamma" test will be ignored when the number of valid nghbr pixels < self.min_nghbr
        self.gamma_exception = gamma_exception

        # padding for depth map
        self.pad = (k + (k - 1) * (d - 1)) // 2

        # index of the center pixel
        self.center_idx = (k*k - 1) // 2

        # torch Unfold to unfold the depth map
        self.unfold = torch.nn.Unfold(kernel_size=(k, k), padding=self.pad, dilation=d)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        """ points: 3D points in camera coordinates, shape: (B, 3, H, W)
        """
        b, _, h, w = points.shape

        # matrix_a (b, h, w, k*k, 3)
        torch_patches = self.unfold(points)                                     # (b, 3*k*k, h, w)
        # import ipdb; ipdb.set_trace()   
        matrix_a = torch_patches.view(b, 3, self.k * self.k, h, w)              # (b, 3, k*k, h, w)
        matrix_a = matrix_a.permute(0, 3, 4, 2, 1)                              # (b, h, w, k*k, 3)
        # filter by depth
        valid_condition = torch.logical_and(points[:,2:,:,:] > self.d_min, points[:,2:,:,:] < self.d_max)
        valid_condition = valid_condition.float()                               # (B, 1, H, W)
        valid_condition = self.unfold(valid_condition)                          # (b, 1*k*k, h, w)
        valid_condition = valid_condition.view(b, 1, self.k * self.k, h, w)     # (b, 1, k*k, h, w)
        valid_condition = valid_condition.permute(0, 3, 4, 2, 1)                # (b, h, w, k*k, 1)

        # valid_condition (b, h, w, k*k, 1)
        if self.gamma is not None:
            valid_depth_diff = torch.abs(matrix_a[:, :, :, :, 2:] - matrix_a[:, :, :, self.center_idx:self.center_idx+1, 2:]) \
                            / matrix_a[:, :, :, self.center_idx:self.center_idx+1, 2:]
            valid_depth_diff = (valid_depth_diff < self.gamma).float()              # (b, h, w, k*k, 1)

            if self.gamma_exception:
                valid_depth_diff_sum = torch.sum(valid_depth_diff, dim=3, keepdim=True)     # (b, h, w, 1, 1)
                valid_depth_diff_sum = (valid_depth_diff_sum < self.min_nghbr).float()     # (b, h, w, 1, 1)    
                valid_depth_diff = valid_depth_diff + valid_depth_diff_sum
                valid_depth_diff = (valid_depth_diff > 0.5).float()

            valid_condition = valid_condition * valid_depth_diff

        # matrix A (b, h, w, k*k, 4)
        matrix_1 = torch.ones_like(matrix_a[:,:,:,:,0:1])
        matrix_A = torch.cat([matrix_a, matrix_1], dim=-1)

        # fill zero for invalid pixels
        matrix_A_zero = torch.zeros_like(matrix_A)
        matrix_A = torch.where(valid_condition.repeat([1, 1, 1, 1, 4]) > 0.5, matrix_A, matrix_A_zero)

        # transpose
        matrix_At = torch.transpose(matrix_A, 3, 4)

        matrix_A = matrix_A.view(-1, self.k * self.k, 4)    # (b*h*w, k*k, 4)
        matrix_At = matrix_At.view(-1, 4, self.k * self.k)  # (b*h*w, 4, k*k)
        At_A = torch.bmm(matrix_At, matrix_A)               # (b*h*w, 4, 4)

        # eig_val: (b*h*w, 4) / eig_vec: (b*h*w, 4, 4)
        eig_val, eig_vec = torch.linalg.eig(At_A)

        # valid_mask (b*h*w)
        valid_eig = torch.logical_and(torch.sum(eig_val.imag, dim=1) == 0,
                        torch.sum(eig_vec.imag, dim=(1, 2)) == 0)

        # find the smallest eigenvalue
        eig_val = eig_val.real
        eig_vec = eig_vec.real

        idx = torch.argmin(eig_val, dim=1, keepdim=True)  # (b*h*w, 1)
        idx_onehot = torch.zeros_like(eig_val).scatter_(1, idx, 1.) # (b*h*w, 4)
        idx_onehot = idx_onehot.unsqueeze(1).repeat(1, 4, 1)

        # normal (b, 3, h, w)
        normal = torch.sum(eig_vec * idx_onehot, dim=2)
        normal = normal.view(b, h, w, 4).permute(0, 3, 1, 2).contiguous()
        normal = F.normalize(normal[:,:3,:,:], p=2.0, dim=1, eps=1e-12)

        # flip if needed
        flip = torch.sign(torch.sum(normal * points, dim=1, keepdim=True))
        normal = normal * flip

        # valid_mask1 (b, 1, h, w): center pixel valid depth
        valid_mask1 = valid_condition[:,:,:,self.center_idx,0].unsqueeze(1)

        # valid_mask2 (b, 1, h, w): sufficient number of valid neighbors
        valid_mask2 = torch.sum(valid_condition[..., 0], dim=3).unsqueeze(1) >= self.min_nghbr

        # valid_mask3 (b, 1, h, w): eigenvalue real
        valid_mask3 = valid_eig.view(b, h, w).unsqueeze(1)

        # valid_mask4 (b, 1, h, w):
        valid_mask4 = torch.norm(normal, p=2, dim=1, keepdim=True) > 0.5

        # valid_mask
        valid_mask = valid_mask1 * valid_mask2 * valid_mask3 * valid_mask4
        
        return normal, valid_mask > 0.5