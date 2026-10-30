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
# Copyright (C) 2025, University of Liege
# TELIM research group, http://www.telecom.ulg.ac.be/
# All rights reserved.
# The modifications are under the LICENSE.md file.
#
# For inquiries contact jan.held@uliege.be
#

from argparse import ArgumentParser, Namespace
import sys
import os

class GroupParams:
    pass

class ParamGroup:
    def __init__(self, parser: ArgumentParser, name : str, fill_none = False):
        group = parser.add_argument_group(name)
        for key, value in vars(self).items():
            shorthand = False
            if key.startswith("_"):
                shorthand = True
                key = key[1:]
            t = type(value)
            value = value if not fill_none else None
            if shorthand:
                if t == bool:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, action="store_true")
                else:
                    group.add_argument("--" + key, ("-" + key[0:1]), default=value, type=t)
            else:
                if t == bool:
                    group.add_argument("--" + key, default=value, action="store_true")
                else:
                    group.add_argument("--" + key, default=value, type=t)

    def extract(self, args):
        group = GroupParams()
        for arg in vars(args).items():
            if arg[0] in vars(self) or ("_" + arg[0]) in vars(self):
                setattr(group, arg[0], arg[1])
        return group

class ModelParams(ParamGroup): 
    def __init__(self, parser, sentinel=False):
        self.sh_degree = 3
        self._source_path = ""
        self._model_path = ""
        self._images = "images"
        self._resolution = -1
        self._white_background = False
        self.data_device = "cuda"
        self.eval = False
        super().__init__(parser, "Loading Parameters", sentinel)

    def extract(self, args):
        g = super().extract(args)
        g.source_path = os.path.abspath(g.source_path)
        return g

class PipelineParams(ParamGroup):
    def __init__(self, parser):
        self.convert_SHs_python = False
        self.compute_cov3D_python = False
        self.depth_ratio = 1.0
        self.debug = False
        super().__init__(parser, "Pipeline Parameters")

class OptimizationParams(ParamGroup):
    def __init__(self, parser):
        self.iterations = 30_000
        self.position_lr_delay_mult = 0.01
        self.position_lr_max_steps = 30_000
        self.lambda_dssim = 0.2

        self.densification_interval = 100 #500 # TODO : Change the value

        self.densify_from_iter = 500
        self.densify_until_iter = 10000

        self.random_background = False
        
        self.feature_lr = 0.0025
        self.max_points = 400000 #4000000

        # Opacity & weight
        self.set_weight = 0.28
        self.weight_lr =  0.05
        self.lambda_weight = 1.9e-06

        # Equilateral loss
        self.lambda_equilateral = 0.0#8

        # Normal loss
        self.iteration_mesh = 5000
        self.lambda_normals = 0.00005
        self.lambda_normals_super = 0.01

        self.add_percentage = 1.1

        self.set_sigma = 1.0

        # Add new triangles or vertices
        self.intervall_add_triangles = 500

        # Prune triangles and vertices
        self.prune_triangles_threshold = 0.235  # TODO : Do ablation with it

        # PARAMETER SECOND STAGE
        self.lr_triangles_points_init = 0.0018

        self.start_opacity_floor = 1000 #5000 # TODO : Changed the value

        self.start_pruning = 600 #4000   # TODO Change the value
        self.sigma_until = 30000
        self.final_opacity_iter = 24000

        self.sigma_start = 0

        self.splitt_large_triangles = 100
        self.start_upsampling = 20000
        self.upscaling_factor = 2

        self.size_probs_zero = 7.5e-05
        self.size_probs_zero_image_space = 0.0

        self.init_point_size = 1.56

        self.prune_size = 1400

        self.lambda_vertex = 0.00025
        self.max_diff_threshold = 0.5
        self.start_vertex_opt = 12000

        self.lamba_depth = 0.05

        self.depth_lambda_init = 0.01
        self.depth_lambda_final = 0.001

        super().__init__(parser, "Optimization Parameters")

def get_combined_args(parser : ArgumentParser):
    cmdlne_string = sys.argv[1:]
    cfgfile_string = "Namespace()"
    args_cmdline = parser.parse_args(cmdlne_string)

    try:
        cfgfilepath = os.path.join(args_cmdline.model_path, "cfg_args")
        print("Looking for config file in", cfgfilepath)
        with open(cfgfilepath) as cfg_file:
            print("Config file found: {}".format(cfgfilepath))
            cfgfile_string = cfg_file.read()
    except TypeError:
        print("Config file not found at")
        pass
    args_cfgfile = eval(cfgfile_string)

    merged_dict = vars(args_cfgfile).copy()
    for k,v in vars(args_cmdline).items():
        if v != None:
            merged_dict[k] = v
    return Namespace(**merged_dict)

def update_indoor(params):
    params.add_percentage = 1.27
    params.densify_from_iter = 1000
    params.densify_until_iter = 10000
    params.feature_lr = 0.004
    params.size_probs_zero = 0.0
    params.splitt_large_triangles = 500
    params.start_pruning = 3000
    params.weight_lr = 0.05
    params.lambda_weight = 0.0
    params.lambda_normals = 0.00001
    params.lambda_normals_super = 0.01
    params.prune_size = 1300
    params.lambda_vertex = 0.00025
    params.depth_lambda_init = 0.0
    params.depth_lambda_final = 0.0
    params.iteration_mesh = 12000
    return params

