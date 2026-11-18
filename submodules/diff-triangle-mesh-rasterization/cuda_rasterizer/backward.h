/*
 * The original code is under the following copyright:
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE_GS.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 * 
 * The modifications of the code are under the following copyright:
 * Copyright (C) 2024, University of Liege, KAUST and University of Oxford
 * TELIM research group, http://www.telecom.ulg.ac.be/
 * IVUL research group, https://ivul.kaust.edu.sa/
 * VGG research group, https://www.robots.ox.ac.uk/~vgg/
 * All rights reserved.
 * The modifications are under the LICENSE.md file.
 *
 * For inquiries contact jan.held@uliege.be
 */

 #ifndef CUDA_RASTERIZER_BACKWARD_H_INCLUDED
 #define CUDA_RASTERIZER_BACKWARD_H_INCLUDED
 
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 namespace BACKWARD
 {
	 void render(
		 const dim3 grid, dim3 block,
		 const uint2* ranges,
		 const uint32_t* point_list,
		 int W, int H,
		 const float* bg_color,
		 const float sigma,
		 const int* triangles_indices,
		 const float2* normals,
		 const float* offsets,
		 const float4* conic_opacity,
		 const float* depths,
		 const float2* means2D,
		 const float2* phi_center,
		 const float2* p_image,
		 const float* vertex_depth,
		 const float* colors,
		 const float* final_Ts,
		 const uint32_t* n_contrib,
		 const float* dL_dpixels,
		 const float* dL_depths,
		 float2* dL_dnormals,
		 float* dL_doffsets,
		 float3* dL_dmean2D,
		 float* dL_dopacity,
		 float* dL_dnormal3D,
		 float* dL_dcolors,
		 float* dL_dpoints2D,
		 float* dL_dvertice_depth);

	 void computeVertexColorGradients(
        int V, int D, int M,
		int W, int H,
		const float* view,
		const float* proj,
		const float* proj_raw, 
		float* p_w,
        const float* vertices,
        const float* shs,
        const bool* clamped,	
        const glm::vec3* campos,
        const float* dL_dcolor,
		const float* dL_dpoints2D,
        glm::vec3* dL_dvertices3D,
        float* dL_dsh,
		const float* dL_dvertice_depth,
		float* dL_dtau);

	 // Pose-only propagation of the barycentric screen and depth gradients.
	 // Must run in both tracking and mapping mode (see backward.cu).
	 void propagateScreenGradsToPose(
		int V,
		const float* viewmatrix,
		const float* proj_raw,
		const float* vertices,
		const float* dL_dpoints2D,
		const float* dL_dvertice_depth,
		int W, int H,
		float* dL_dtau);

	 void preprocess(
		 int P, int D, int M,
		 const float* vertices,
		 const int* triangles_indices,
		 const float* vertex_weights,
		 int W, int H,
		 const int* radii,
		 const float* shs,
		 const bool* clamped,
		 const float* view,
		 const float* proj,
		 const float* proj_raw,  
		 float2* points_xy_image,
		 float* p_w,
		 float2* p_image,
		 int* indices,
		 const float focal_x, float focal_y,
		 const float tan_fovx, float tan_fovy,
		 const glm::vec3* campos,
		 glm::vec3* dL_dvertices3D,
		 float* dL_dvertice_weights,
		 const float2* dL_dnormals,
		 const float* dL_doffsets,
		 float3* dL_dmean2D,
		 float* dL_dopacity,
		 float* dL_dnormal3D,
		 float* dL_dcolor,
		 float* dL_dsh,
		 float* dL_dtau,
		 bool optimize_pose_only = false,
		 bool opacity_min = false
		 );
 }
 
 #endif