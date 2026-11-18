import torch
from mesh_splatting.utils.loss_utils import ssim

# Cached Scharr filter kernels — allocated once per device on first use.
_SLAM_KERNELS: dict = {}

def _get_scharr_kernels(device):
    key = str(device)
    if key not in _SLAM_KERNELS:
        y = torch.tensor([[3, 0, -3], [10, 0, -10], [3, 0, -3]], dtype=torch.float32, device=device)
        x = torch.tensor([[3, 10, 3], [0, 0, 0], [-3, -10, -3]], dtype=torch.float32, device=device)
        ones = torch.ones((1, 1, 3, 3), dtype=torch.float32, device=device)
        _SLAM_KERNELS[key] = {"y": y, "x": x, "norm": 1.0 / y.abs().sum(), "ones": ones}
    return _SLAM_KERNELS[key]


def image_gradient(image):
    # Compute image gradient using Scharr Filter
    c = image.shape[0]
    k = _get_scharr_kernels(image.device)
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    img_grad_v = k["norm"] * torch.nn.functional.conv2d(
        p_img, k["x"].view(1, 1, 3, 3).expand(c, 1, 3, 3), groups=c
    )
    img_grad_h = k["norm"] * torch.nn.functional.conv2d(
        p_img, k["y"].view(1, 1, 3, 3).expand(c, 1, 3, 3), groups=c
    )
    return img_grad_v[0], img_grad_h[0]


def image_gradient_mask(image, eps=0.01):
    # Compute image gradient mask
    c = image.shape[0]
    k = _get_scharr_kernels(image.device)
    ones = k["ones"]
    p_img = torch.nn.functional.pad(image, (1, 1, 1, 1), mode="reflect")[None]
    p_img = torch.abs(p_img) > eps
    img_grad_v = torch.nn.functional.conv2d(
        p_img.float(), ones.expand(c, 1, 3, 3), groups=c
    )
    img_grad_h = torch.nn.functional.conv2d(
        p_img.float(), ones.expand(c, 1, 3, 3), groups=c
    )

    return img_grad_v[0] == 9.0, img_grad_h[0] == 9.0


def depth_reg(depth, gt_image, huber_eps=0.1, mask=None):
    mask_v, mask_h = image_gradient_mask(depth)
    gray_grad_v, gray_grad_h = image_gradient(gt_image.mean(dim=0, keepdim=True))
    depth_grad_v, depth_grad_h = image_gradient(depth)
    gray_grad_v, gray_grad_h = gray_grad_v[mask_v], gray_grad_h[mask_h]
    depth_grad_v, depth_grad_h = depth_grad_v[mask_v], depth_grad_h[mask_h]

    w_h = torch.exp(-10 * gray_grad_h**2)
    w_v = torch.exp(-10 * gray_grad_v**2)
    err = (w_h * torch.abs(depth_grad_h)).mean() + (
        w_v * torch.abs(depth_grad_v)
    ).mean()
    return err


def get_loss_tracking_test(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    monocular = config["Dataset"]["sensor_type"] == "monocular"
    # return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)

    # if monocular:
        # print("Using RGB loss")
    return get_loss_tracking_rgb_test(config, image_ab, depth, opacity, viewpoint)
    # print("Using RGBD loss")
    #TODO: fix depth loss tracking
    # return get_depth_loss(config, opacity, depth, viewpoint) #+ get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    # return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb_test(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1 = opacity * torch.abs(image - gt_image)
    return l1.mean(), l1


def get_loss_tracking(config, image, depth, opacity, viewpoint, initialization=False):
    image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b

    monocular = config["Dataset"]["sensor_type"] == "monocular"
    # return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)

    if monocular:
        # print("Using RGB loss")
        return get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    # print("Using RGBD loss")
    #TODO: fix depth loss tracking
    # return get_depth_loss(config, opacity, depth, viewpoint) #+ get_loss_tracking_rgb(config, image_ab, depth, opacity, viewpoint)
    return get_loss_tracking_rgbd(config, image_ab, depth, opacity, viewpoint)


def get_loss_tracking_rgb(config, image, depth, opacity, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    l1 = opacity * torch.abs(image * viewpoint.rgb_pixel_mask - 
                             gt_image * viewpoint.rgb_pixel_mask)
    return l1.mean(), l1

def get_depth_loss(config, opacity, depth, viewpoint):
    gt_depth = viewpoint.depth if isinstance(viewpoint.depth, torch.Tensor) else torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype)
    depth_pixel_mask = (gt_depth > 0.01).reshape(depth.shape)
    opacity_mask = (opacity > config["Training"]["alpha_opacity"]).reshape(depth.shape)
    depth_mask = depth_pixel_mask * opacity_mask
    gt_depth_shaped = gt_depth.reshape(depth.shape)
    l1_depth = torch.abs(depth * depth_mask - gt_depth_shaped * depth_mask)
    return l1_depth.mean()

def get_loss_tracking_rgbd(
    config, image, depth, opacity, viewpoint, initialization=False
):
    rgb_weight = config["Training"]["rgb_weight"] if "rgb_weight" in config["Training"] else 0.9
    max_depth = config["Training"].get("max_depth", 0)
    # viewpoint.depth may be numpy; ensure masks and gt_depth are tensors on same device as depth
    gt_depth = viewpoint.depth if isinstance(viewpoint.depth, torch.Tensor) else torch.as_tensor(viewpoint.depth, device=depth.device, dtype=depth.dtype)
    depth_pixel_mask = (gt_depth > 0.01).reshape(depth.shape)
    if max_depth > 0:
        depth_pixel_mask = depth_pixel_mask & (gt_depth.reshape(depth.shape) < max_depth)
    opacity_mask = (opacity > 0.95).reshape(depth.shape)

    l1_rgb, l1 = get_loss_tracking_rgb(config, image, depth, opacity, viewpoint)
    depth_mask = depth_pixel_mask * opacity_mask
    gt_depth_shaped = gt_depth.reshape(depth.shape)
    l1_depth = torch.abs(depth * depth_mask - gt_depth_shaped * depth_mask)
    return rgb_weight * l1_rgb + (1 - rgb_weight) * l1_depth.mean(), l1


def get_loss_mapping(config, image, depth, viewpoint, opacity, initialization=False):
    if initialization:
        image_ab = image
    else:
        image_ab = (torch.exp(viewpoint.exposure_a)) * image + viewpoint.exposure_b
    if config["Dataset"]["sensor_type"] == "monocular":
        return get_loss_mapping_rgb(config, image_ab, depth, viewpoint)
    return get_loss_mapping_rgbd(config, image_ab, depth, viewpoint)


def get_loss_mapping_rgb(config, image, depth, viewpoint):
    gt_image = viewpoint.original_image.cuda()
    mask = viewpoint.rgb_pixel_mask_mapping.cuda()

    l1_rgb = torch.abs(image * mask - gt_image * mask).mean()

    lambda_dssim = config.get("opt_params", {}).get("lambda_dssim", 0.0)
    if lambda_dssim > 0.0:
        try:
            from fused_ssim import fused_ssim
            ssim_value = fused_ssim((image * mask).unsqueeze(0), (gt_image * mask).unsqueeze(0))
        except Exception:
            ssim_value = ssim(image * mask, gt_image * mask)
        return (1.0 - lambda_dssim) * l1_rgb + lambda_dssim * (1.0 - ssim_value)

    return l1_rgb


def get_loss_mapping_rgbd(config, image, depth, viewpoint, initialization=False):
    rgb_weight = config["Training"]["rgb_weight"] if "rgb_weight" in config["Training"] else 0.9
    max_depth = config["Training"].get("max_depth", 0)
    mask_sky_rgb = config["Training"].get("mask_sky_rgb", False)
    device = image.device

    # Use GPU-preloaded tensors when available (set by mapping loop to avoid per-iter CPU->GPU transfers).
    gt_image       = getattr(viewpoint, "_gt_image_gpu",  viewpoint.original_image).to(device)
    rgb_pixel_mask = getattr(viewpoint, "_rgb_mask_gpu",  viewpoint.rgb_pixel_mask_mapping).to(device)
    gt_depth       = getattr(viewpoint, "_gt_depth_gpu",  viewpoint.gt_depth).to(device)

    depth_pixel_mask = (gt_depth > 0.01).view(*depth.shape)
    if max_depth > 0:
        depth_pixel_mask = depth_pixel_mask & (gt_depth.view(*depth.shape) < max_depth)

    if mask_sky_rgb:
        sky_mask = (gt_depth.view(*depth.shape) < 0.01)
        if max_depth > 0:
            sky_mask = sky_mask | (gt_depth.view(*depth.shape) >= max_depth)
        rgb_pixel_mask = rgb_pixel_mask & ~sky_mask

    l1_rgb = torch.abs(image * rgb_pixel_mask - gt_image * rgb_pixel_mask)
    l1_depth = torch.abs(depth * depth_pixel_mask - gt_depth * depth_pixel_mask)
    l1_rgb_mean = l1_rgb.mean()
    l1_depth_mean = l1_depth.mean()

    lambda_dssim = config.get("opt_params", {}).get("lambda_dssim", 0.0)
    if lambda_dssim > 0.0:
        try:
            from fused_ssim import fused_ssim
            ssim_value = fused_ssim((image * rgb_pixel_mask).unsqueeze(0), (gt_image * rgb_pixel_mask).unsqueeze(0))
        except Exception:
            ssim_value = ssim(image * rgb_pixel_mask, gt_image * rgb_pixel_mask)
        photo_loss = (1.0 - lambda_dssim) * l1_rgb_mean + lambda_dssim * (1.0 - ssim_value)
    else:
        photo_loss = l1_rgb_mean

    return rgb_weight * photo_loss + (1 - rgb_weight) * l1_depth_mean


def get_median_depth(depth, opacity=None, mask=None, return_std=False):
    depth = depth.detach().clone()
    opacity = opacity.detach()
    valid = depth > 0
    if opacity is not None:
        valid = torch.logical_and(valid, opacity > 0.95)
    if mask is not None:
        valid = torch.logical_and(valid, mask)
    valid_depth = depth[valid]
    if return_std:
        return valid_depth.median(), valid_depth.std(), valid
    return valid_depth.median()
