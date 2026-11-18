import sys
from pathlib import Path

_root = Path(__file__).resolve().parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from utils import keyframe_buffer

from utils.normal_utils import Depth2normal

import contextlib
import copy
import time
from argparse import ArgumentParser
import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
# import rerun as rr
import torch

#==================================================
# Imported Libraries
#==================================================
# import config
# from event_datasets.dataset import get_dataset
#===================================================

from mesh_splatting.scene.triangle_model import TriangleModel
from mesh_splatting.utils.graphics_utils import getProjectionMatrix2, getWorld2View2
from mesh_splatting.utils.sh_utils import eval_sh
from mesh_splatting.triangle_renderer import render

from utils.config_utils import load_config
from utils.dataset import load_dataset
from utils.gui_utils import rerun_show_cam, log_trajectories_rerun, log_mesh_rerun
from utils.cameras import Camera
from utils.pose_utils import SE3_exp
from utils.keyframe_buffer import KeyframeBuffer
from utils.eval_utils import compute_ate_se3_from_cameras, eval_ate, Log, eval_rendering

from utils.profiler import SLAMProfiler
from mesh_splatting.arguments import ModelParams, PipelineParams, OptimizationParams

from meshslam.backend import MapState, MappingEngine
from meshslam.frontend import TrackingEngine

import json
import os
import yaml
from datetime import datetime
import wandb
from pathlib import Path
from mesh_splatting.utils.system_utils import mkdir_p

import multiprocessing as mp
from meshslam.gui.gui_utils import TrianglePacket, ParamsGUI, FakeQueue



from utils.normal_utils import normal_to_rgb

from fused_ssim import fused_ssim

from mesh_splatting.utils.sh_utils import RGB2SH
from simple_knn._C import distCUDA2


# config = load_config("event_configs/SimuEvent/replica_room0.yaml")
# print(config)
  