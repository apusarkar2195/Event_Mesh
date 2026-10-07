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

#ifndef CUDA_RASTERIZER_H_INCLUDED
#define CUDA_RASTERIZER_H_INCLUDED

#include <vector>
#include <functional>

namespace CudaRasterizer
{
	class Rasterizer
	{
	public:

		static void markVisible(
			int P,
			float* means3D,
			float* viewmatrix,
			float* projmatrix,
			bool* present);

		static int forward(
			std::function<char* (size_t)> geometryBuffer,
			std::function<char* (size_t)> binningBuffer,
			std::function<char* (size_t)> imageBuffer,
			const int P, const int V, int D, int M,
			const float* background,
			const int width, int height,
			const float* vertices,
			const int* triangles_indices,
			const float* vertex_weights,
			const float sigma,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			float* scaling,
			const float* viewmatrix,
			const float* projmatrix,
			const float* cam_pos,
			const float tan_fovx, float tan_fovy,
			const bool prefiltered,
			float* out_color,
			float* out_others,
			float* max_blending,
			int* radii = nullptr,
			int* was_rendered = nullptr,
			bool debug = false,
			float min_triangle_area_pixels = 0.5f,
			bool opacity_min = false);

		static void backward(
			const int P, const int V, int D, int M, int R,
			const float* background,
			const int width, int height,
			const float* proj_raw,
			const float* vertices,
			const int* triangles_indices,
			const float* vertex_weights,
			const float sigma,
			const int total_nb_points,
			const float* shs,
			const float* colors_precomp,
			const float* viewmatrix,
			const float* projmatrix,
			const float* campos,
			const float tan_fovx, float tan_fovy,
			const int* radii,
			char* geom_buffer,
			char* binning_buffer,
			char* image_buffer,
			const float* dL_dpix,
			const float* dL_depths,
			float* dL_dmeans2D,
			float* dL_dnormal3D,
			float* dL_dvertices3D,
			float* dL_dvertex_weights,
			float* dL_dnormals,
			float* dL_doffsets,
			float* dL_dopacity,
			float* dL_dcolor,
			float* dL_dsh,
			float* dL_dpoints2D,
			float* dL_dvertice_depth,
			float* dL_dtau,
			bool debug,
			bool optimize_pose_only = false,
			bool opacity_min = false);
	};
};

#endif