import os
os.environ['MPLBACKEND'] = 'Agg'
import numpy as np



import matplotlib
matplotlib.use("Agg", force=True)  # Set backend before importing evo.tools.plot
# Monkey-patch to prevent backend switching
_original_use = matplotlib.use
def _patched_use(name, force=False):
    if name in ['TkAgg', 'TKAgg']:
        return _original_use('Agg', force=True)
    return _original_use(name, force=force)
matplotlib.use = _patched_use
import matplotlib.pyplot as plt


import evo
from evo.core import metrics, trajectory
from evo.tools import plot
from evo.core.trajectory import PosePath3D, PoseTrajectory3D
import os
import json
import wandb
from mesh_splatting.utils.system_utils import mkdir_p
import rich
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
import torch
import cv2

from mesh_splatting.triangle_renderer import render
from mesh_splatting.render_utils import get_psnr
from mesh_splatting.utils.loss_utils import ssim



_log_styles = {
    "TSSLAM": "bold green",
    "GUI": "bold magenta",
    "Eval": "bold red",
    "MonoGS": "bold yellow",
}


def get_style(tag):
    if tag in _log_styles.keys():
        return _log_styles[tag]
    return "bold blue"



def Log(*args, tag="TSSLAM"):
    style = get_style(tag)
    rich.print(f"[{style}]{tag}:[/{style}]", *args)


def compute_ate_se3(gt_centers, tracked_centers):
    """ATE (m) with SE(3) alignment (Umeyama). Both (N, 3)."""
    gt = np.asarray(gt_centers, dtype=np.float64)
    est = np.asarray(tracked_centers, dtype=np.float64)
    n = gt.shape[0]
    mean_gt = gt.mean(axis=0)
    mean_est = est.mean(axis=0)
    gt_centered = gt - mean_gt
    est_centered = est - mean_est
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mean_gt - R @ mean_est
    aligned_est = (est @ R.T) + t
    errors = np.linalg.norm(gt - aligned_est, axis=1)
    ate_rmse = np.sqrt(np.mean(errors ** 2))
    return ate_rmse, errors

def compute_ate_se3_from_cameras(gt_cameras, est_cameras, match_by_uid=True):
    """
    ATE (m) with SE(3) alignment from two lists of cameras.

    - gt_cameras: list of Camera (GT); uses .camera_center_gt and .uid.
    - est_cameras: list of Camera (estimated); uses .camera_center and .uid.
    - match_by_uid: if True, pair by .uid (est_cameras can be a subset); if False, use same order and length.
    """
    if match_by_uid:
        gt_by_uid = {getattr(c, "uid", i): c for i, c in enumerate(gt_cameras)}
        gt_centers, est_centers = [], []
        for c_est in est_cameras:
            uid = getattr(c_est, "uid", None)
            if uid is None or uid not in gt_by_uid:
                continue
            c_gt = gt_by_uid[uid]
            gt_centers.append(c_gt.camera_center_gt.detach().cpu().numpy())
            est_centers.append(c_est.camera_center.detach().cpu().numpy())
        if not gt_centers:
            return float("nan"), np.array([])
        gt_centers = np.array(gt_centers, dtype=np.float64)
        est_centers = np.array(est_centers, dtype=np.float64)
    else:
        n = min(len(gt_cameras), len(est_cameras))
        gt_centers = np.array(
            [gt_cameras[i].camera_center_gt.detach().cpu().numpy() for i in range(n)],
            dtype=np.float64,
        )
        est_centers = np.array(
            [est_cameras[i].camera_center.detach().cpu().numpy() for i in range(n)],
            dtype=np.float64,
        )
    return compute_ate_se3(gt_centers, est_centers)


def umeyama_se3(est_centers, gt_centers):
    """
    Umeyama SE(3) alignment: est -> gt.
    Returns (R, t) so that aligned_est = (est @ R.T) + t is in gt frame.
    Both inputs (N, 3). Returns R (3,3), t (3,).
    """
    est = np.asarray(est_centers, dtype=np.float64)
    gt = np.asarray(gt_centers, dtype=np.float64)
    mean_gt = gt.mean(axis=0)
    mean_est = est.mean(axis=0)
    gt_centered = gt - mean_gt
    est_centered = est - mean_est
    H = est_centered.T @ gt_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T
    if np.linalg.det(R) < 0:
        Vt[-1] *= -1
        R = Vt.T @ U.T
    t = mean_gt - R @ mean_est
    return R, t

def evaluate_evo(poses_gt, poses_est, plot_dir, label, monocular=False, plot_axes="xy"):
    ## Plot
    traj_ref = PosePath3D(poses_se3=poses_gt)
    traj_est = PosePath3D(poses_se3=poses_est)

    if traj_ref.num_poses < 3 or traj_est.num_poses < 3:
        return 0.0

           
    traj_est_aligned = PosePath3D(poses_se3=poses_est)
    traj_est_aligned.align(traj_ref, correct_scale=monocular)


    ## RMSE
    pose_relation = metrics.PoseRelation.translation_part
    data = (traj_ref, traj_est_aligned)
    ape_metric = metrics.APE(pose_relation)
    ape_metric.process_data(data)
    ape_stat = ape_metric.get_statistic(metrics.StatisticsType.rmse)
    ape_stats = ape_metric.get_all_statistics()
    Log("RMSE ATE \[m]", ape_stat, tag="Eval")

    with open(
        os.path.join(plot_dir, "stats_{}.json".format(str(label))),
        "w",
        encoding="utf-8",
    ) as f:
        json.dump(ape_stats, f, indent=4)

    plot_mode = evo.tools.plot.PlotMode[plot_axes]
    fig = plt.figure()
    ax = evo.tools.plot.prepare_axis(fig, plot_mode)
    ax.set_title(f"ATE RMSE: {ape_stat}")
    evo.tools.plot.traj(ax, plot_mode, traj_ref, "--", "gray", "gt")
    evo.tools.plot.traj_colormap(
        ax,
        traj_est_aligned,
        ape_metric.error,
        plot_mode,
        min_map=ape_stats["min"],
        max_map=ape_stats["max"],
    )
    ax.legend()
    plt.savefig(os.path.join(plot_dir, "evo_2dplot_{}.png".format(str(label))), dpi=90)
    plt.close(fig)

    return ape_stat


def eval_ate(frames, kf_ids, save_dir, iterations, final=False, monocular=False, plot_axes="xy"):
    trj_data = dict()
    latest_frame_idx = kf_ids[-1] + 2 if final else kf_ids[-1] + 1
    trj_id, trj_est, trj_gt = [], [], []
    trj_est_np, trj_gt_np = [], []


    for kf_id in kf_ids:
        kf = frames[kf_id]
        pose_est = np.linalg.inv(kf.world_view_transform.T.cpu().numpy())
        pose_gt = np.linalg.inv(kf.world_view_transform_gt.T.cpu().numpy())

        trj_id.append(frames[kf_id].uid)
        trj_est.append(pose_est.tolist())
        trj_gt.append(pose_gt.tolist())

        trj_est_np.append(pose_est)
        trj_gt_np.append(pose_gt)

    trj_data["trj_id"] = trj_id
    trj_data["trj_est"] = trj_est
    trj_data["trj_gt"] = trj_gt

    plot_dir = os.path.join(save_dir, "plot")
    mkdir_p(plot_dir)

    label_evo = "final" if final else "{:04}".format(iterations)
    with open(
        os.path.join(plot_dir, f"trj_{label_evo}.json"), "w", encoding="utf-8"
    ) as f:
        json.dump(trj_data, f, indent=4)

    ate = evaluate_evo(
        poses_gt=trj_gt_np,
        poses_est=trj_est_np,
        plot_dir=plot_dir,
        label=label_evo,
        monocular=monocular,
        plot_axes=plot_axes,
    )
    if wandb.run is not None:
        wandb.log({"frame_idx": latest_frame_idx, "ate": ate})
    return ate




def eval_rendering(
    frames,
    triangles,
    dataset,
    save_dir,
    pipe,
    background,
    kf_indices,
    iteration="final",
):
    interval = 5
    img_pred, img_gt, saved_frame_idx = [], [], []
    end_idx = len(frames) - 1 if iteration == "final" or "before_opt" else iteration
    psnr_array, ssim_array, lpips_array = [], [], []
    cal_lpips = LearnedPerceptualImagePatchSimilarity(
        net_type="alex", normalize=True
    ).to("cuda")
    for idx in range(0, end_idx, interval):
        if idx in kf_indices:
            continue
        saved_frame_idx.append(idx)
        frame = frames[idx]
        gt_image, _, _ = dataset[idx]

        rendering = render(frame, triangles, pipe, background)["render"]
        image = torch.clamp(rendering, 0.0, 1.0)

        gt = (gt_image.cpu().numpy().transpose((1, 2, 0)) * 255).astype(np.uint8)
        pred = (image.detach().cpu().numpy().transpose((1, 2, 0)) * 255).astype(
            np.uint8
        )
        gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
        pred = cv2.cvtColor(pred, cv2.COLOR_BGR2RGB)
        img_pred.append(pred)
        img_gt.append(gt)

        mask = gt_image > 0

        psnr_score = get_psnr((image[mask]).unsqueeze(0), (gt_image[mask]).unsqueeze(0))
        ssim_score = ssim((image).unsqueeze(0), (gt_image).unsqueeze(0))
        lpips_score = cal_lpips((image).unsqueeze(0), (gt_image).unsqueeze(0))

        psnr_array.append(psnr_score.item())
        ssim_array.append(ssim_score.item())
        lpips_array.append(lpips_score.item())

    output = dict()
    output["mean_psnr"] = float(np.mean(psnr_array))
    output["mean_ssim"] = float(np.mean(ssim_array))
    output["mean_lpips"] = float(np.mean(lpips_array))

    Log(
        f'mean psnr: {output["mean_psnr"]}, ssim: {output["mean_ssim"]}, lpips: {output["mean_lpips"]}',
        tag="Eval",
    )

    psnr_save_dir = os.path.join(save_dir, "psnr", str(iteration))
    mkdir_p(psnr_save_dir)

    json.dump(
        output,
        open(os.path.join(psnr_save_dir, "final_result.json"), "w", encoding="utf-8"),
        indent=4,
    )
    return output
