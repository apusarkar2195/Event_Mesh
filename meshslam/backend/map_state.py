"""
Map state: triangle model, background, keyframe buffer.
Holds everything the backend owns; designed so it can later be serialized
or passed to another process for multi-process SLAM.
"""
from dataclasses import dataclass, field
from typing import Optional

import torch

from mesh_splatting.scene.triangle_model import TriangleModel
from utils.keyframe_buffer import KeyframeBuffer
import os

@dataclass
class MapState:
    """State owned by the backend (mapping + keyframes)."""
    triangles: TriangleModel
    background: torch.Tensor
    keyframe_buffer: KeyframeBuffer
    device: torch.device = torch.device("cuda")
    model_path: str = ""
    # Optional: filled by MappingEngine for reporting 
    timings: dict = field(default_factory=dict)
    loss_history: list = field(default_factory=list)

    def get_triangles(self):
        """Return the triangle model (for rendering in frontend or elsewhere)."""
        return self.triangles

    def get_background(self):
        """Return background tensor for rendering."""
        return self.background

    def get_keyframe_buffer(self):
        """Return keyframe buffer (e.g. for replay sampling)."""
        return self.keyframe_buffer


    def save_mesh(self, iteration):
        point_cloud_path = os.path.join(self.model_path, "mesh/iteration_{}".format(iteration))
        print("Save model to: ", point_cloud_path)
        self.triangles.save_parameters(point_cloud_path)
        # Save SLAM keyframe poses so offline refinement can reuse optimized SLAM views.
        keyframes = list(self.keyframe_buffer)
        keyframe_state = []
        for kf in keyframes:
            keyframe_state.append(
                {
                    "uid": int(kf.uid),
                    "R": kf.R.detach().cpu(),
                    "T": kf.T.detach().cpu(),
                }
            )
        torch.save(
            {"keyframes": keyframe_state},
            os.path.join(point_cloud_path, "keyframes_state.pt"),
        )

        return os.path.join(point_cloud_path, 'point_cloud_state_dict.pt')

