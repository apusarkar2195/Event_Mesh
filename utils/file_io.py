
import os
import cv2
import glob

import torch
import math

def read_groundtruth_poses(groundtruth_file):
    groundtruth_poses = []

    with open(groundtruth_file, 'r') as file:
        for line in file:
            if line.startswith('#'):
                continue
            
            # Parse the line
            parts = line.strip().split()
            if len(parts) == 8:  # Ensure the line has the correct number of elements
                timestamp = float(parts[0])
                tx, ty, tz = map(float, parts[1:4])
                qx, qy, qz, qw = map(float, parts[4:])
                
                # Append to the list as a dictionary
                groundtruth_poses.append({
                    "timestamp": timestamp,
                    "translation": (tx, ty, tz),
                    "quaternion": (qx, qy, qz, qw)
                })

    return groundtruth_poses


# Functions to select frames by timestamps/ids
# Example usage: 
# frame_index1, frame_index2 = select_frames_by_indices(0, 10)  # Select by indices
# frame_index1, frame_index2 = select_frames_by_timestamps(groundtruth_poses[0]["timestamp"], groundtruth_poses[10]["timestamp"])  # Select by timestamps
def select_frames_by_indices(rgb_images, depth_images, index1, index2):
    print(len(depth_images), len(rgb_images))
    if index1 < 0 or index2 < 0 or index1 >= len(depth_images) or index2 >= len(depth_images):
        raise ValueError("Indices are out of range.")
    return index1, index2

def select_frames_by_timestamps(groundtruth_poses, timestamp1, timestamp2):
    timestamps = [pose["timestamp"] for pose in groundtruth_poses]
    if timestamp1 not in timestamps or timestamp2 not in timestamps:
        raise ValueError("Timestamps not found in ground truth poses.")
    index1 = timestamps.index(timestamp1)
    index2 = timestamps.index(timestamp2)
    return index1, index2


def read_timestamp_file(timestamp_file, folder):
    entries = []

    with open(timestamp_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue

            ts, fname = line.strip().split()
            img_path = os.path.join(folder, fname)
            # print(f"Checking image path: {folder}, {fname}")  # Debugging line
            if os.path.exists(img_path):
               entries.append((float(ts), img_path))

    entries.sort(key=lambda x: x[0])
    timestamps = [e[0] for e in entries]
    images = [e[1] for e in entries]
    return timestamps, images

def load_images(folder, rgb_timestamp_file, depth_timestamp_file, n):
    """
    Load RGB and depth images from the specified timestamp files.
    """
    # Read RGB and depth images with timestamps
    rgb_timestamps, image_files_rgb = read_timestamp_file(rgb_timestamp_file, folder)
    depth_timestamps, image_files_depth = read_timestamp_file(depth_timestamp_file, folder)

    # TODO: align the timestamps of RGB and depth images
    # For now, we assume they are already aligned and just take the first n images
    
    images_rgb = [cv2.imread(f) for f in image_files_rgb[17:(17+n)]]
    images_depth = [cv2.imread(f, cv2.IMREAD_UNCHANGED) for f in image_files_depth[:n]]

    return images_rgb, images_depth

import numpy as np
import trimesh
class TUMParser:
    def __init__(self, input_folder):
        self.input_folder = input_folder
        self.load_poses(self.input_folder, frame_rate=32)
        self.n_img = len(self.color_paths)

    def parse_list(self, filepath, skiprows=0):
        data = np.loadtxt(filepath, delimiter=" ", dtype=str, skiprows=skiprows)
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

            frame = {
                "file_path": str(os.path.join(datapath, image_data[i, 1])),
                "depth_path": str(os.path.join(datapath, depth_data[j, 1])),
                "transform_matrix": (np.linalg.inv(T)).tolist(),
            }

            self.frames.append(frame)


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


def focal2fov(focal, pixels):
    return 2 * math.atan(pixels / (2 * focal))

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
            # if self.disorted:
            #     depth = cv2.remap(depth, self.map1x, self.map1y, cv2.INTER_NEAREST)

        image = (
            torch.from_numpy(image / 255.0)
            .clamp(0.0, 1.0)
            .permute(2, 0, 1)
            .to(device=self.device, dtype=self.dtype)
        )
        pose = torch.from_numpy(pose).to(device=self.device)
        return image, depth, pose


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
            pose = np.linalg.inv(pose)
            self.poses.append(pose)
            frame = {
                "file_path": self.color_paths[i],
                "depth_path": self.depth_paths[i],
                "transform_matrix": pose.tolist(),
            }

            frames.append(frame)
        self.frames = frames



class TUMDataset(MonocularDataset):
    def __init__(self, args, path, config):
        super().__init__(args, path, config)
        parser = TUMParser(path)
        self.num_imgs = parser.n_img
        self.color_paths = parser.color_paths
        self.depth_paths = parser.depth_paths
        self.poses = parser.poses
