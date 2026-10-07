import csv
import glob
import os

import cv2
import numpy as np
import torch
import trimesh


from mesh_splatting.utils.graphics_utils import focal2fov

try:
    import pyrealsense2 as rs
except Exception:
    rs = None

try:
    import pyk4a
    from pyk4a import PyK4A, Config as K4AConfig, ColorResolution, DepthMode, FPS, CalibrationType
except Exception:
    pyk4a = None


class KITTIOdometryParser:
    """Parser for KITTI Odometry dataset with pre-computed depth maps.

    Expected layout:
        base_path/
        ├── poses/XX.txt              (T_w_cam0, 12 floats per line)
        └── sequences/XX/
            ├── calib.txt
            ├── image_2/000000.png    (color left)
            ├── image_3/000000.png    (color right)
            └── depth/000000.npy      (pre-computed metric depth)
    """

    def __init__(self, input_folder, sequence="07"):
        self.input_folder = input_folder
        self.sequence = sequence
        seq_dir = os.path.join(input_folder, "sequences", sequence)

        self.color_paths = sorted(glob.glob(f"{seq_dir}/image_2/*.png"))
        self.depth_paths = sorted(glob.glob(f"{seq_dir}/depth/*.npy"))
        self.n_img = len(self.color_paths)

        # Parse calibration
        self.calib = self._load_calib(os.path.join(seq_dir, "calib.txt"))

        # Parse poses (T_w_cam0 -> T_cam2_w)
        pose_file = os.path.join(input_folder, "poses", f"{sequence}.txt")
        self.load_poses(pose_file)

    def _load_calib(self, path):
        calib = {}
        with open(path, "r") as f:
            for line in f:
                key, *vals = line.strip().split()
                key = key.rstrip(":")
                calib[key] = np.array([float(v) for v in vals]).reshape(3, 4)
        return calib

    def _get_intrinsics(self):
        P2 = self.calib["P2"]
        fx, fy = P2[0, 0], P2[1, 1]
        cx, cy = P2[0, 2], P2[1, 2]
        return fx, fy, cx, cy

    def _get_stereo_baseline(self):
        """Baseline between color cameras (cam2 and cam3) in meters."""
        P2 = self.calib["P2"]
        P3 = self.calib["P3"]
        fx = P2[0, 0]
        return abs(P3[0, 3] - P2[0, 3]) / fx

    def load_poses(self, path):
        self.poses = []
        self.frames = []
        P2 = self.calib["P2"]
        fx = P2[0, 0]

        # T_cam0_cam2: transform from cam2 to cam0 (small baseline offset)
        K = np.array([[fx, 0, P2[0, 2]], [0, P2[1, 1], P2[1, 2]], [0, 0, 1]])
        t_cam0_cam2 = np.linalg.solve(K, P2[:, 3])
        T_cam0_cam2 = np.eye(4)
        T_cam0_cam2[:3, 3] = t_cam0_cam2

        with open(path, "r") as f:
            lines = f.readlines()

        for i in range(min(len(lines), self.n_img)):
            vals = list(map(float, lines[i].strip().split()))
            T_w_cam0 = np.eye(4)
            T_w_cam0[:3, :] = np.array(vals).reshape(3, 4)

            # Convert to cam2 frame and invert to get T_cam2_w
            T_w_cam2 = T_w_cam0 @ T_cam0_cam2
            T_cam2_w = np.linalg.inv(T_w_cam2)
            self.poses.append(T_cam2_w)

            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i] if i < len(self.depth_paths) else None,
                "transform_matrix": T_cam2_w.tolist(),
            }
            self.frames.append(frame)


class OxfordSpiresParser:
    """Parser for prepared Oxford Spires dataset.

    Expected layout (produced by scripts/prepare_oxford_spires.py):
        base_path/
        ├── images/000000.jpg         (undistorted pinhole images)
        ├── depth/000000.npy          (z-depth maps, float32, meters)
        ├── poses.txt                 (TUM format: timestamp tx ty tz qx qy qz qw)
        └── calibration.json          (pinhole intrinsics)
    """

    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{input_folder}/images/*.jpg"))
        if not self.color_paths:
            self.color_paths = sorted(glob.glob(f"{input_folder}/images/*.png"))
        self.depth_paths = sorted(glob.glob(f"{input_folder}/depth/*.npy"))
        self.n_img = len(self.color_paths)
        self.load_poses(os.path.join(input_folder, "poses.txt"))

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                tx, ty, tz = float(parts[1]), float(parts[2]), float(parts[3])
                qx, qy, qz, qw = float(parts[4]), float(parts[5]), float(parts[6]), float(parts[7])
                from scipy.spatial.transform import Rotation
                T_world_cam = np.eye(4)
                T_world_cam[:3, :3] = Rotation.from_quat([qx, qy, qz, qw]).as_matrix()
                T_world_cam[:3, 3] = [tx, ty, tz]
                # SLAM expects T_cam_world (w2c)
                self.poses.append(np.linalg.inv(T_world_cam))


class ReplicaParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.color_paths = sorted(glob.glob(f"{self.input_folder}/results/frame*.jpg"))
        self.depth_paths = sorted(glob.glob(f"{self.input_folder}/results/depth*.png"))
        self.n_img = len(self.color_paths)
        self.load_poses(f"{self.input_folder}/traj.txt")

    def load_poses(self, path):
        self.poses = []
        with open(path, "r") as f:
            lines = f.readlines()

        frames = []
        for i in range(self.n_img):
            line = lines[i]
            pose = np.array(list(map(float, line.split()))).reshape(4, 4)
            pose = np.linalg.inv(pose)  # SLAM expect the w2c camera pose. The GT pose is c2w
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames


class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=np.str_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if np.abs(tstamp_depth[j] - t) < max_dt:
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and (
                    np.abs(tstamp_pose[k] - t) < max_dt
                ):
                    associations.append((i, j, k))

        return associations

    def load_poses(self, datapath, frame_rate=-1):
        if os.path.isfile(os.path.join(datapath, "groundtruth.txt")):
            pose_list = os.path.join(datapath, "groundtruth.txt")
        elif os.path.isfile(os.path.join(datapath, "pose.txt")):
            pose_list = os.path.join(datapath, "pose.txt")

        image_list = os.path.join(datapath, "rgb.txt")
        depth_list = os.path.join(datapath, "depth.txt")

        image_data = self.parse_list(image_list)
        depth_data = self.parse_list(depth_list)
        pose_data = self.parse_list(pose_list, skiprows=1)
        pose_vecs = pose_data[:, 0:].astype(np.float64)

        tstamp_image = image_data[:, 0].astype(np.float64)
        tstamp_depth = depth_data[:, 0].astype(np.float64)
        tstamp_pose = pose_data[:, 0].astype(np.float64)
        associations = self.associate_frames(tstamp_image, tstamp_depth, tstamp_pose)

        indicies = [0]
        for i in range(1, len(associations)):
            t0 = tstamp_image[associations[indicies[-1]][0]]
            t1 = tstamp_image[associations[i][0]]
            if t1 - t0 > 1.0 / frame_rate:
                indicies += [i]

        self.color_paths, self.poses, self.depth_paths, self.frames = [], [], [], []

        for ix in indicies:
            (i, j, k) = associations[ix]
            self.color_paths += [os.path.join(datapath, image_data[i, 1])]
            self.depth_paths += [os.path.join(datapath, depth_data[j, 1])]

            quat = pose_vecs[k][4:]
            trans = pose_vecs[k][1:4]
            T = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T[:3, 3] = trans

            self.poses += [np.linalg.inv(T)]
            # print(self.poses)

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


class EuRoCParser:
    def __init__(self, input_folder, start_idx=0):
        self.input_folder = input_folder
        self.start_idx = start_idx
        self.color_paths = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam0/data/*.png")
        )
        self.color_paths_r = sorted(
            glob.glob(f"{self.input_folder}/mav0/cam1/data/*.png")
        )
        assert len(self.color_paths) == len(self.color_paths_r)
        self.color_paths = self.color_paths[start_idx:]
        self.color_paths_r = self.color_paths_r[start_idx:]
        self.n_img = len(self.color_paths)
        self.load_poses(
            f"{self.input_folder}/mav0/state_groundtruth_estimate0/data.csv"
        )

    def associate(self, ts_pose):
        pose_indices = []
        for i in range(self.n_img):
            color_ts = float((self.color_paths[i].split("/")[-1]).split(".")[0])
            k = np.argmin(np.abs(ts_pose - color_ts))
            pose_indices.append(k)

        return pose_indices

    def load_poses(self, path):
        self.poses = []
        with open(path) as f:
            reader = csv.reader(f)
            header = next(reader)
            data = [list(map(float, row)) for row in reader]
        data = np.array(data)
        T_i_c0 = np.array(
            [
                [0.0148655429818, -0.999880929698, 0.00414029679422, -0.0216401454975],
                [0.999557249008, 0.0149672133247, 0.025715529948, -0.064676986768],
                [-0.0257744366974, 0.00375618835797, 0.999660727178, 0.00981073058949],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        pose_ts = data[:, 0]
        pose_indices = self.associate(pose_ts)

        frames = []
        
        
        for i in range(self.n_img):
            trans = data[pose_indices[i], 1:4]
            quat = data[pose_indices[i], 4:8]

            T_w_i = trimesh.transformations.quaternion_matrix(np.roll(quat, 1))
            T_w_i[:3, 3] = trans
            T_w_c = np.dot(T_w_i, T_i_c0)
            self.poses += [np.linalg.inv(T_w_c)]

            frame = {
                "file_path": self.color_paths[i],
                "transform_matrix": (np.linalg.inv(T_w_c)).tolist(),
            }

            frames.append(frame)
        self.frames = frames


class _LazyIdentityPoses:
    """Lightweight stub for live datasets that have no ground-truth poses.
    Returns an identity 4x4 matrix for any index without pre-allocating memory."""
    def __init__(self, n=999999):
        self._n = n

    def __len__(self):
        return self._n

    def __getitem__(self, idx):
        return np.eye(4, dtype=np.float64)


class BaseDataset(torch.utils.data.Dataset):
    def __init__(self, args, path, config):
        self.args = args
        self.path = path
        self.config = config
        self.device = "cuda:0"
        self.dtype = torch.float32
        self.num_imgs = 999999

    def __len__(self):
        return self.num_imgs

    def __getitem__(self, idx):
        pass


class MonocularDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        # Camera prameters
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )
        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [
                calibration["k1"],
                calibration["k2"],
                calibration["p1"],
                calibration["p2"],
                calibration["k3"],
            ]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K,
            self.dist_coeffs,
            np.eye(3),
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )
        # depth parameters
        self.has_depth = True if "depth_scale" in calibration.keys() else False
        self.depth_scale = calibration["depth_scale"] if self.has_depth else None

        # Default scene scale
        nerf_normalization_radius = 5
        self.scene_info = {
            "nerf_normalization": {
                "radius": nerf_normalization_radius,
                "translation": np.zeros(3),
            },
        }

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)
        depth = None

        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        if self.has_depth:
            depth_path = self.depth_paths[idx]
            depth = cv2.imread(depth_path, cv2.IMREAD_ANYDEPTH) / self.depth_scale
            


        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class StereoDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.width = calibration["width"]
        self.height = calibration["height"]

        cam0raw = calibration["cam0"]["raw"]
        cam0opt = calibration["cam0"]["opt"]
        cam1raw = calibration["cam1"]["raw"]
        cam1opt = calibration["cam1"]["opt"]
        # Camera prameters
        self.fx_raw = cam0raw["fx"]
        self.fy_raw = cam0raw["fy"]
        self.cx_raw = cam0raw["cx"]
        self.cy_raw = cam0raw["cy"]
        self.fx = cam0opt["fx"]
        self.fy = cam0opt["fy"]
        self.cx = cam0opt["cx"]
        self.cy = cam0opt["cy"]

        self.fx_raw_r = cam1raw["fx"]
        self.fy_raw_r = cam1raw["fy"]
        self.cx_raw_r = cam1raw["cx"]
        self.cy_raw_r = cam1raw["cy"]
        self.fx_r = cam1opt["fx"]
        self.fy_r = cam1opt["fy"]
        self.cx_r = cam1opt["cx"]
        self.cy_r = cam1opt["cy"]

        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K_raw = np.array(
            [
                [self.fx_raw, 0.0, self.cx_raw],
                [0.0, self.fy_raw, self.cy_raw],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.Rmat = np.array(calibration["cam0"]["R"]["data"]).reshape(3, 3)
        self.K_raw_r = np.array(
            [
                [self.fx_raw_r, 0.0, self.cx_raw_r],
                [0.0, self.fy_raw_r, self.cy_raw_r],
                [0.0, 0.0, 1.0],
            ]
        )

        self.K_r = np.array(
            [[self.fx_r, 0.0, self.cx_r], [0.0, self.fy_r, self.cy_r], [0.0, 0.0, 1.0]]
        )
        self.Rmat_r = np.array(calibration["cam1"]["R"]["data"]).reshape(3, 3)

        # distortion parameters
        self.disorted = calibration["distorted"]
        self.dist_coeffs = np.array(
            [cam0raw["k1"], cam0raw["k2"], cam0raw["p1"], cam0raw["p2"], cam0raw["k3"]]
        )
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K_raw,
            self.dist_coeffs,
            self.Rmat,
            self.K,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

        self.dist_coeffs_r = np.array(
            [cam1raw["k1"], cam1raw["k2"], cam1raw["p1"], cam1raw["p2"], cam1raw["k3"]]
        )
        self.map1x_r, self.map1y_r = cv2.initUndistortRectifyMap(
            self.K_raw_r,
            self.dist_coeffs_r,
            self.Rmat_r,
            self.K_r,
            (self.width, self.height),
            cv2.CV_32FC1,
        )

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        color_path_r = self.color_paths_r[idx]

        pose = self.poses[idx]
        image = cv2.imread(color_path, 0)
        image_r = cv2.imread(color_path_r, 0)
        depth = None
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
            image_r = cv2.remap(image_r, self.map1x_r, self.map1y_r, cv2.INTER_LINEAR)
        stereo = cv2.StereoSGBM_create(minDisparity=0, numDisparities=64, blockSize=20)
        stereo.setUniquenessRatio(40)
        disparity = stereo.compute(image, image_r) / 16.0
        disparity[disparity == 0] = 1e10
        depth = 47.90639384423901 / (
            disparity
        )  ## Following ORB-SLAM2 config, baseline*fx
        depth[depth < 0] = 0
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)

        return image, depth, pose


class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        parser = TUMParser(path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths   
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class ReplicaDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        parser = ReplicaParser(path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses


class KITTIDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        sequence = config["Dataset"].get("sequence", "07")
        parser = KITTIOdometryParser(path, sequence=sequence)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)

        depth = None
        if idx < len(self.depth_paths) and self.depth_paths[idx] is not None:
            depth = np.load(self.depth_paths[idx]).astype(np.float32)
            depth[depth < 0] = 0
            depth[~np.isfinite(depth)] = 0

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class OxfordSpiresDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        parser = OxfordSpiresParser(path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
        self.max_depth = config["Dataset"].get("max_depth", 80.0)
        self._depth_is_metric = config["Dataset"].get("depth_is_metric", True)
        self._downsample = config["Dataset"].get("image_downsample", 1)
        if self._downsample > 1:
            self.width = self.width // self._downsample
            self.height = self.height // self._downsample
            self.fx = self.fx / self._downsample
            self.fy = self.fy / self._downsample
            self.cx = self.cx / self._downsample
            self.cy = self.cy / self._downsample
            self.fovx = focal2fov(self.fx, self.width)
            self.fovy = focal2fov(self.fy, self.height)
            self.K = np.array(
                [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
            )
            print(f"OxfordSpires: downsampling {self._downsample}x -> {self.width}x{self.height}")

    def __getitem__(self, idx):
        color_path = self.color_paths[idx]
        pose = self.poses[idx]
        image = cv2.cvtColor(cv2.imread(color_path), cv2.COLOR_BGR2RGB)

        depth = None
        if idx < len(self.depth_paths) and self.depth_paths[idx] is not None:
            depth = np.load(self.depth_paths[idx]).astype(np.float32)
            depth[depth < 0] = 0
            depth[~np.isfinite(depth)] = 0
            depth[depth > self.max_depth] = 0

        if self._downsample > 1:
            h, w = image.shape[:2]
            new_w, new_h = w // self._downsample, h // self._downsample
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
            if depth is not None:
                depth = cv2.resize(depth, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


class EurocDataset(StereoDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        parser = EuRoCParser(path, start_idx=0)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.color_paths_r = parser.color_paths_r
        self.poses = parser.poses


class RealsenseDataset(BaseDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        if rs is None:
            raise ImportError(
                "pyrealsense2 is required for live RealSense. Install with: pip install pyrealsense2"
            )
        self.pipeline = rs.pipeline()
        self.h, self.w = 360, 640
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, self.w, self.h, rs.format.bgr8, 30)
        self.config.enable_stream(rs.stream.depth)
        self.profile = self.pipeline.start(self.config)
        self.align_to = rs.stream.color
        self.align = rs.align(self.align_to)
        
        self.rgb_sensor = self.profile.get_device().query_sensors()[1]
        self.rgb_sensor.set_option(rs.option.enable_auto_exposure, False)
        self.rgb_sensor.set_option(rs.option.enable_auto_white_balance, False)
        self.rgb_sensor.set_option(rs.option.exposure, 100)
        self.rgb_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.color)
        )

        self.rgb_intrinsics = self.rgb_profile.get_intrinsics()


        
        
        self.fx = self.rgb_intrinsics.fx
        self.fy = self.rgb_intrinsics.fy
        self.cx = self.rgb_intrinsics.ppx
        self.cy = self.rgb_intrinsics.ppy
        self.width = self.rgb_intrinsics.width
        self.height = self.rgb_intrinsics.height
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]]
        )

        self.disorted = True
        self.dist_coeffs = np.asarray(self.rgb_intrinsics.coeffs)
        self.map1x, self.map1y = cv2.initUndistortRectifyMap(
            self.K, self.dist_coeffs, np.eye(3), self.K, (self.w, self.h), cv2.CV_32FC1
        )

        # Stub poses so slam.py's len(dataset.poses) works for live mode.
        self.poses = _LazyIdentityPoses()

        # depth parameters
        self.has_depth = config["Dataset"]["sensor_type"] == "depth"
        self.num_frames = 0

        if self.has_depth:
            self.depth_sensor = self.profile.get_device().first_depth_sensor()
            self.depth_scale  = self.depth_sensor.get_depth_scale()
            self.depth_profile = rs.video_stream_profile(
            self.profile.get_stream(rs.stream.depth)
            )
            self.depth_intrinsics = self.depth_profile.get_intrinsics()
            print("Depth Scale is: " , self.depth_scale)
            print("Depth intrinsics: ", self.depth_intrinsics)
            print("RGB intrinsics: ", self.rgb_intrinsics)

        # Flush warmup frames so the pipeline is streaming reliably before SLAM reads.
        # The first ~30 frames often don't arrive within the default 5 s timeout
        # because the camera needs auto-exposure settle time after start().
        print("Warming up RealSense camera (flushing 30 frames)...")
        for i in range(30):
            try:
                self.pipeline.wait_for_frames(timeout_ms=10000)
            except RuntimeError:
                # Camera occasionally fails to deliver a frame; retry after a short pause.
                import time as _time
                print(f"  warmup frame {i} timed out, retrying...")
                _time.sleep(0.5)
                try:
                    self.pipeline.wait_for_frames(timeout_ms=10000)
                except RuntimeError:
                    print(f"  warmup frame {i} retry also timed out, restarting pipeline...")
                    self.pipeline.stop()
                    _time.sleep(1.0)
                    self.profile = self.pipeline.start(self.config)
                    continue
        print("RealSense camera ready.")
            

    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)

        frameset = self.pipeline.wait_for_frames(timeout_ms=10000)
        aligned_frames = self.align.process(frameset)
        self.num_frames = self.num_frames + 1


        rgb_frame = aligned_frames.get_color_frame()
        image = np.asanyarray(rgb_frame.get_data())
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        
        depth = None
        if self.has_depth:
            aligned_depth_frame = aligned_frames.get_depth_frame()
            depth = np.array(aligned_depth_frame.get_data())*self.depth_scale
            depth[depth < 0] = 0
            np.nan_to_num(depth, nan=1000)

        return image, depth, pose

class KinectDataset(BaseDataset):
    """Live Azure Kinect DK RGB-D camera via pyk4a."""

    # Maps config string → (width, height)
    _COLOR_RES = {
        "RES_720P":  (1280,  720),
        "RES_1080P": (1920, 1080),
        "RES_1440P": (2560, 1440),
        "RES_1536P": (2048, 1536),
        "RES_2160P": (3840, 2160),
        "RES_3072P": (4096, 3072),
    }

    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        if pyk4a is None:
            raise ImportError(
                "pyk4a is required for Azure Kinect. Install with: pip install pyk4a"
            )

        dataset_cfg = config["Dataset"]
        self.has_depth = dataset_cfg.get("sensor_type", "depth") == "depth"

        color_res_str = dataset_cfg.get("color_resolution", "RES_720P")
        color_resolution = getattr(ColorResolution, color_res_str)

        fps_val = dataset_cfg.get("fps", 30)
        fps = getattr(FPS, f"FPS_{fps_val}")

        depth_mode_str = dataset_cfg.get("depth_mode", "NFOV_UNBINNED")
        depth_mode = getattr(DepthMode, depth_mode_str) if self.has_depth else DepthMode.OFF

        self.k4a = PyK4A(K4AConfig(
            color_resolution=color_resolution,
            depth_mode=depth_mode,
            camera_fps=fps,
            synchronized_images_only=self.has_depth,
        ))
        self.k4a.start()

        # Intrinsics from hardware calibration
        cam_matrix = self.k4a.calibration.get_camera_matrix(CalibrationType.COLOR)
        self.fx = float(cam_matrix[0, 0])
        self.fy = float(cam_matrix[1, 1])
        self.cx = float(cam_matrix[0, 2])
        self.cy = float(cam_matrix[1, 2])

        self.width, self.height = self._COLOR_RES.get(color_res_str, (1280, 720))
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0., 0., 1.]])

        # Azure Kinect color sensor has low distortion; skip correction by default
        self.disorted = False

        # Depth is returned in mm by transformed_depth; convert to metres
        self.depth_scale = 1000.0

        self.poses = _LazyIdentityPoses()

        print("Warming up Azure Kinect (flushing 30 frames)...")
        for _ in range(30):
            self.k4a.get_capture()
        print("Azure Kinect ready.")

    def __getitem__(self, idx):
        pose = torch.eye(4, device=self.device, dtype=self.dtype)

        capture = self.k4a.get_capture()

        # capture.color is BGRA uint8 (H, W, 4)
        image = cv2.cvtColor(capture.color[..., :3], cv2.COLOR_BGR2RGB)
        image = (torch.from_numpy(image / 255.0)
                 .clamp(0.0, 1.0)
                 .permute(2, 0, 1)
                 .to(device=self.device, dtype=self.dtype))

        depth = None
        if self.has_depth:
            # transformed_depth: uint16 mm, aligned to color camera space
            depth = capture.transformed_depth.astype(np.float32) / self.depth_scale
            depth[depth <= 0] = 0

        return image, depth, pose


class WebcamDataset(BaseDataset):
    """Live RGB camera via OpenCV VideoCapture. Monocular (no depth)."""

    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        calibration = config["Dataset"]["Calibration"]
        self.fx = calibration["fx"]
        self.fy = calibration["fy"]
        self.cx = calibration["cx"]
        self.cy = calibration["cy"]
        self.width  = calibration["width"]
        self.height = calibration["height"]
        self.fovx = focal2fov(self.fx, self.width)
        self.fovy = focal2fov(self.fy, self.height)
        self.K = np.array([[self.fx, 0., self.cx],
                           [0., self.fy, self.cy],
                           [0., 0., 1.]])

        # Optional lens distortion
        self.disorted = calibration.get("distorted", False)
        self.dist_coeffs = np.array([
            calibration.get("k1", 0.), calibration.get("k2", 0.),
            calibration.get("p1", 0.), calibration.get("p2", 0.),
            calibration.get("k3", 0.),
        ])
        if self.disorted:
            self.map1x, self.map1y = cv2.initUndistortRectifyMap(
                self.K, self.dist_coeffs, np.eye(3), self.K,
                (self.width, self.height), cv2.CV_32FC1)

        cam_idx = config["Dataset"].get("camera_index", 0)
        self.cap = cv2.VideoCapture(cam_idx)
        if not self.cap.isOpened():
            raise RuntimeError(f"Cannot open camera index {cam_idx}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)

        # Stub poses so slam.py's len(dataset.poses) works for live mode.
        self.poses = _LazyIdentityPoses()

    def __getitem__(self, idx):
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Failed to capture frame from webcam")
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if self.disorted:
            image = cv2.remap(image, self.map1x, self.map1y, cv2.INTER_LINEAR)
        image = (torch.from_numpy(image / 255.0)
                 .clamp(0.0, 1.0)
                 .permute(2, 0, 1)
                 .to(device=self.device, dtype=self.dtype))
        pose = torch.eye(4, device=self.device, dtype=self.dtype)
        return image, None, pose  # depth=None → monocular mode


def load_dataset(args, path, config):
    if config["Dataset"]["type"] == "tum":
        return TUMDataset(args, path, config)
    elif config["Dataset"]["type"] == "replica":
        return ReplicaDataset(args, path, config)
    elif config["Dataset"]["type"] == "euroc":
        return EurocDataset(args, path, config)
    elif config["Dataset"]["type"] == "realsense":
        return RealsenseDataset(args, path, config)
    elif config["Dataset"]["type"] == "webcam":
        return WebcamDataset(args, path, config)
    elif config["Dataset"]["type"] == "kinect":
        return KinectDataset(args, path, config)
    elif config["Dataset"]["type"] == "kitti":
        return KITTIDataset(args, path, config)
    elif config["Dataset"]["type"] == "oxford_spires":
        return OxfordSpiresDataset(args, path, config)
    else:
        raise ValueError("Unknown dataset type")
