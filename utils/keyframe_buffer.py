import copy
from multiprocessing import reduction
import random
from collections import deque
import torch

class KeyframeBuffer:
    def __init__(self, maxlen, device="cuda"):
        self._buffer = deque(maxlen=maxlen) if maxlen else deque()
        self._by_uid = {}  # uid -> keyframe for O(1) lookup
        self.occ_aware_visibility = {}
        self._device = device

    def _move_keyframe_heavy_to_cpu(self, entry):
        """Move heavy read-only keyframe data to CPU to save GPU memory."""
        attrs_to_move = [
            "original_image", "depth", "normal", 
            "grad_mask", "rgb_pixel_mask", "rgb_pixel_mask_mapping", 
            "gt_depth", "mask"
        ]
        for attr in attrs_to_move:
            val = getattr(entry, attr, None)
            if val is not None and hasattr(val, "cpu"):
                setattr(entry, attr, val.cpu())
                
        # Also clean up raw numpy arrays if they aren't strictly needed
        if hasattr(entry, "normal_raw"):
            entry.normal_raw = None

    def add(self, camera):
        # Store a clone so we don't mutate the live camera
        entry = copy.deepcopy(camera)
        # Ensure pose is fixed for replay (no grad)
        if hasattr(entry, 'R'):
            entry.R = entry.R.detach().clone()
        if hasattr(entry, 'T'):
            entry.T = entry.T.detach().clone()

        # Evict oldest from index when buffer is full (deque will drop it on append)
        if self._buffer.maxlen and len(self._buffer) == self._buffer.maxlen:
            old_uid = getattr(self._buffer[0], "uid", None)
            if old_uid is not None:
                self._by_uid.pop(old_uid, None)

        self._move_keyframe_heavy_to_cpu(entry)

        self._buffer.append(entry)
        uid = getattr(entry, "uid", None)
        if uid is not None:
            self._by_uid[uid] = entry

    def add_with_visibility(self, camera, render_pkg):
        # Store a clone so we don't mutate the live camera
        entry = copy.deepcopy(camera)
        # Ensure pose is fixed for replay (no grad)
        if hasattr(entry, 'R'):
            entry.R = entry.R.detach().clone()
        if hasattr(entry, 'T'):
            entry.T = entry.T.detach().clone()

        # Evict oldest from index when buffer is full (deque will drop it on append)
        if self._buffer.maxlen and len(self._buffer) == self._buffer.maxlen:
            old_uid = getattr(self._buffer[0], "uid", None)
            if old_uid is not None:
                self._by_uid.pop(old_uid, None)

        self._move_keyframe_heavy_to_cpu(entry)

        self._buffer.append(entry)
        uid = getattr(entry, "uid", None)
        if uid is not None:
            self._by_uid[uid] = entry

        self.occ_aware_visibility[uid] = render_pkg["curr_visibility"]

    def update_visibilities(self, render_callback):
        """Re-render each keyframe and update its stored visibility. render_callback(kf) must return a dict with 'curr_visibility'."""
        for kf in list(self._buffer):
            render_pkg_kf = render_callback(kf)
            if render_pkg_kf and "curr_visibility" in render_pkg_kf:
                uid = getattr(kf, "uid", None)
                if uid is not None:
                    self.occ_aware_visibility[uid] = render_pkg_kf["curr_visibility"]

    def __iter__(self):
        return iter(self._buffer)

    def get_by_uid(self, uid):
        """Return the keyframe camera with this camera.uid, or None if not in buffer."""
        return self._by_uid.get(uid)

    def get_keyframes_ids(self, slice_start=0, slice_end=None):
        slice_start = min(slice_start, len(self._buffer))
        if slice_end is None:
            slice_end = len(self._buffer)
        return [kf.uid for kf in list(self._buffer)[slice_start:slice_end]]

    def get_visibility(self, uid):
        return self.occ_aware_visibility.get(uid)

    def update_visibility(self, uid, visibility):
        self.occ_aware_visibility[uid] = visibility

    def get_replay_sample(self, n):
        if len(self._buffer) == 0:
            return []
        k = min(n, len(self._buffer))
        indices = random.sample(range(len(self._buffer)), k)
        return [self._buffer[i] for i in indices]

    def __len__(self):
        return len(self._buffer)

    def get_last_keyframe(self):
        """Return the last keyframe camera in the buffer."""
        if len(self._buffer) == 0:
            return None
        return self._buffer[-1]

    def get_penultimate_keyframe(self):
        """Return the penultimate keyframe camera in the buffer."""
        return self._buffer[-2]

    def get_last_keyframe_visibility(self):
        """Return the visibility of the last keyframe."""
        return self.occ_aware_visibility.get(self.get_last_keyframe().uid)

    def get_penultimate_keyframe_visibility(self):
        """Return the visibility of the penultimate keyframe."""
        return self.occ_aware_visibility.get(self.get_penultimate_keyframe().uid)


    def get_replay_sample_with_visibility(self, n, current_visibility=None):
        if len(self._buffer) == 0:
            return []
        k = min(n, len(self._buffer))
        
        if current_visibility is None:
            # Fallback: random sampling (current behavior)
            indices = random.sample(range(len(self._buffer)), k)
            return [self._buffer[i] for i in indices]
        
        # Compute covisibility for each keyframe with current view
        scores = []
        for kf in self._buffer:
            kf_vis = self.occ_aware_visibility.get(getattr(kf, "uid", None))
            if kf_vis is None:
                scores.append(0.0)
                continue
            # Ensure same device and compatible dtypes for logical ops
            curr = current_visibility.bool() if current_visibility.dtype != torch.bool else current_visibility
            kf_v = kf_vis.bool() if kf_vis.dtype != torch.bool else kf_vis
            if curr.device != kf_v.device:
                kf_v = kf_v.to(curr.device)
            inter = torch.logical_and(curr, kf_v).count_nonzero().item()
            uni = torch.logical_or(curr, kf_v).count_nonzero().item()
            covis = inter / uni if uni > 0 else 0.0
            scores.append(covis)
        
        # Sort by covisibility descending, take top k
        indices = sorted(range(len(self._buffer)), key=lambda i: scores[i], reverse=True)[:k]
        return [self._buffer[i] for i in indices]


    def get_replay_sample_with_visibility_and_distance(
        self,
        n,
        current_visibility=None,
        current_camera_center=None,
        use_covisibility=True,
        use_distance=False,
        covis_weight=1.0,
        dist_weight=1.0,
    ):

        if len(self._buffer) == 0:
            return []

        if n < 3:
            return list(self._buffer)

        k = min(n, len(self._buffer))
        
        if current_visibility is None or current_camera_center is None:
            indices = random.sample(range(len(self._buffer)), k)
            return [self._buffer[i] for i in indices]
        
        scores = []
        for kf in self._buffer:
            # Covisibility
            kf_vis = self.occ_aware_visibility.get(getattr(kf, "uid", None))
            if kf_vis is None:
                covis = 0.0
            else:
                curr = current_visibility.bool().to(kf_vis.device)
                kf_v = kf_vis.bool()
                # NOTE: Be very careful here. Make sure to prune all the visibilities when we do pruning on the map.
                # densiifcation is fine because triangles are always added at the end
                assert curr.shape == kf_v.shape, "Current visibility and keyframe visibility must have the same shape"
                curr = curr.flatten()
                kf_v = kf_v.flatten()
                inter = torch.logical_and(curr, kf_v).count_nonzero().item()
                uni = torch.logical_or(curr, kf_v).count_nonzero().item()
                covis = inter / uni if uni > 0 else 0.0
            
            # Distance (furthest = larger)
            dist = torch.norm(current_camera_center - kf.camera_center).item()
            if use_covisibility and use_distance:
                score = covis_weight * covis + dist_weight * dist
            elif use_covisibility:
                score = covis
            elif use_distance:
                score = dist
            else:
                score = random.random()
        
            
            # Combined: prefer high covisibility AND far (e.g. product)
            score = covis * dist
            scores.append((score, covis, dist))
        
        # Sort by score descending (high covisibility * far distance first)
        indices = sorted(range(len(self._buffer)), key=lambda i: scores[i][0], reverse=True)[:k]
        return [self._buffer[i] for i in indices]


    def pad_visibilities(self, new_tri_count: int):
        for uid, vis in self.occ_aware_visibility.items():
            if vis.numel() < new_tri_count:
                pad = torch.zeros(new_tri_count - vis.numel(), device=vis.device, dtype=vis.dtype)
                self.occ_aware_visibility[uid] = torch.cat([vis, pad], dim=0)

    def apply_prune_mask(self, keep_mask: torch.Tensor):
        for uid, vis in self.occ_aware_visibility.items():
            n = min(vis.numel(), keep_mask.numel())
            self.occ_aware_visibility[uid] = vis[:n][keep_mask[:n]]

