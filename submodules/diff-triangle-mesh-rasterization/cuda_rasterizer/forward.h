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

 #ifndef CUDA_RASTERIZER_FORWARD_H_INCLUDED
 #define CUDA_RASTERIZER_FORWARD_H_INCLUDED
 
 #include <cuda.h>
 #include "cuda_runtime.h"
 #include "device_launch_parameters.h"
 #define GLM_FORCE_CUDA
 #include <glm/glm.hpp>
 
 namespace FORWARD
 {
	 // Perform initial steps for each Triangle prior to rasterization.
	 void preprocess(int P, int D, int M,
		 const float* vertices,
		 const int* triangles_indices,
		 const float* vertex_weights,
		 const float sigma,
		 float* scaling,
		 const float* shs,
		 bool* clamped,
		 const float* colors_precomp,
		 const float* viewmatrix,
		 const float* projmatrix,
		 const glm::vec3* cam_pos,
		 const int W, int H,
		 const float focal_x, float focal_y,
		 const float tan_fovx, float tan_fovy,
		 int* radii,
		 float2* normals,
		 float* offsets,
		 float* p_w,
		 float2* p_image,
		 int* indices,
		 float2* points_xy_image,
		 float* depths,
		 float4* conic_opacity,
		 float2* phi_center,
		 uint2* rect_min,
		 uint2* rect_max,
		 const dim3 grid,
		 uint32_t* tiles_touched,
		 bool prefiltered,
		 float min_triangle_area_pixels,
		 bool opacity_min);

	void computeVertexColors(
		int V, int D, int M,
		const float* vertices,
		const float* shs,
		bool* clamped,
		float* rgb,
		float* vertex_depth, 
		const float* viewmatrix,
		const glm::vec3* cam_pos);
 
	 // Main rasterization method.
	 void render(
		 const dim3 grid, dim3 block,
		 const uint2* ranges,
		 const uint32_t* point_list,
		 int W, int H,
		 const float2* normals,
		 const float* offsets,
		 const float2* points_xy_image,
		 const float* vertex_depth, 
		 const int* triangles_indices,
		 const float sigma,
		 const float* features,
		 const float4* conic_opacity,
		 const float* depths,
		 const float2* phi_center,
		 const float2* p_image,
		 float* final_T,
		 uint32_t* n_contrib,
		 const float* bg_color,
		 float* out_color,
		 float* out_others, 
		 float* max_blending,
		 int* was_rendered);
 }
 
 
 #endif