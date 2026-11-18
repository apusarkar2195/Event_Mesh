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

#include <math.h>
#include <torch/extension.h>
#include <cstdio>
#include <sstream>
#include <iostream>
#include <tuple>
#include <stdio.h>
#include <cuda_runtime_api.h>
#include <memory>
#include "cuda_rasterizer/config.h"
#include "cuda_rasterizer/rasterizer.h"
#include <fstream>
#include <string>
#include <functional>
#include "cuda_rasterizer/utils.h"
#include "cuda_rasterizer/adam.h"

std::function<char*(size_t N)> resizeFunctional(torch::Tensor& t) {
    auto lambda = [&t](size_t N) {
        t.resize_({(long long)N});
		return reinterpret_cast<char*>(t.contiguous().data_ptr());
    };
    return lambda;
}

std::tuple<int, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
RasterizetrianglesCUDA(
	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
	const float sigma,
    const torch::Tensor& colors,
	torch::Tensor& scaling,
	const torch::Tensor& viewmatrix,
	const torch::Tensor& projmatrix,
	const torch::Tensor& projmatrix_raw,
	const float cx,
	const float cy,
	const float tan_fovx, 
	const float tan_fovy,
    const int image_height,
    const int image_width,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const bool prefiltered,
	const bool debug,
	const float min_triangle_area_pixels,
	const bool opacity_min)
{
    
  const int P = triangles_indices.size(0);
  const int V = vertices.size(0); 
  const int H = image_height;
  const int W = image_width;

  auto int_opts = vertices.options().dtype(torch::kInt32);
  auto float_opts = vertices.options().dtype(torch::kFloat32);

  torch::Tensor out_color = torch::full({NUM_CHANNELS, H, W}, 0.0, float_opts);
  torch::Tensor radii = torch::full({P}, 0, vertices.options().dtype(torch::kInt32));

  torch::Tensor proba_existence = torch::full({P}, 0.0, float_opts);
  torch::Tensor was_rendered = torch::full({P}, 0, vertices.options().dtype(torch::kInt32));
  
  torch::Device device(torch::kCUDA);
  torch::TensorOptions options(torch::kByte);
  torch::Tensor geomBuffer = torch::empty({0}, options.device(device));
  torch::Tensor binningBuffer = torch::empty({0}, options.device(device));
  torch::Tensor imgBuffer = torch::empty({0}, options.device(device));
  std::function<char*(size_t)> geomFunc = resizeFunctional(geomBuffer);
  std::function<char*(size_t)> binningFunc = resizeFunctional(binningBuffer);
  std::function<char*(size_t)> imgFunc = resizeFunctional(imgBuffer);

  torch::Tensor out_others = torch::full({3+3+1, H, W}, 0.0, float_opts);
  torch::Tensor max_blending = torch::full({P}, 0.0, float_opts);

  const int total_nb_points = P * 3; // FOR EACH TRIANGLE, WE CAN HAVE 3 NORMALS, OFFSETS,...
  
  int rendered = 0;
  if(P != 0)
  {
	  int M = 0;
	  if(sh.size(0) != 0)
	  {
		M = sh.size(1);
      }

	  rendered = CudaRasterizer::Rasterizer::forward(
	    geomFunc,
		binningFunc,
		imgFunc,
	    P, V, degree, M,
		background.contiguous().data_ptr<float>(),
		W, H,
		vertices.contiguous().data_ptr<float>(),
		triangles_indices.contiguous().data_ptr<int>(),
		vertex_weights.contiguous().data_ptr<float>(),
		sigma,
		total_nb_points,
		sh.contiguous().data_ptr<float>(),
		colors.contiguous().data_ptr<float>(), 
		scaling.contiguous().data_ptr<float>(), 
		viewmatrix.contiguous().data_ptr<float>(), 
		projmatrix.contiguous().data_ptr<float>(),
		campos.contiguous().data_ptr<float>(),
		tan_fovx,
		tan_fovy,
		prefiltered,
		out_color.contiguous().data_ptr<float>(),
		out_others.contiguous().data_ptr<float>(),
		max_blending.contiguous().data_ptr<float>(),
		radii.contiguous().data_ptr<int>(),
		was_rendered.contiguous().data_ptr<int>(),
		debug,
		min_triangle_area_pixels,
		opacity_min);
  }
  return std::make_tuple(rendered, out_color, out_others, radii, was_rendered, geomBuffer, binningBuffer, imgBuffer, scaling, max_blending);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>
 RasterizetrianglesBackwardCUDA(
 	const torch::Tensor& background,
	const torch::Tensor& vertices,
	const torch::Tensor& triangles_indices,
	const torch::Tensor& vertex_weights,
    const float sigma,
	const torch::Tensor& radii,
    const torch::Tensor& colors,
	const torch::Tensor& viewmatrix,
    const torch::Tensor& projmatrix,
	const torch::Tensor& projmatrix_raw,
	const float cx,
	const float cy,
	const float tan_fovx,
	const float tan_fovy,
    const torch::Tensor& dL_dout_color,
	const torch::Tensor& dL_dout_others,
	const torch::Tensor& sh,
	const int degree,
	const torch::Tensor& campos,
	const torch::Tensor& geomBuffer,
	const int R,
	const torch::Tensor& binningBuffer,
	const torch::Tensor& imageBuffer,
	const bool debug,
	const bool optimize_pose_only,
	const bool opacity_min) 
{
  const int P = triangles_indices.size(0); // number of triangles
  const int H = dL_dout_color.size(1);
  const int W = dL_dout_color.size(2);
  
  int M = 0;
  if(sh.size(0) != 0)
  {	
	M = sh.size(1);
  }

  const int V = vertices.size(0); // total number of vertices
  const int total_nb_points = P * 3;

  torch::Tensor dL_dvertices3D = torch::zeros({V, 3}, vertices.options());
  torch::Tensor dL_dvertice_weight = torch::zeros({V}, vertices.options());
  torch::Tensor dL_dpoints2D = torch::zeros({V, 2}, vertices.options());

  torch::Tensor dL_dnormals = torch::zeros({total_nb_points, 3}, vertices.options());
  torch::Tensor dL_doffsets = torch::zeros({total_nb_points, 3}, vertices.options());

  torch::Tensor dL_dcolors = torch::zeros({V, NUM_CHANNELS}, vertices.options());
  torch::Tensor dL_dopacity = torch::zeros({P, 1}, vertices.options());
  torch::Tensor dL_dsh = torch::zeros({V, M, 3}, vertices.options());

  torch::Tensor dL_dmeans2D = torch::zeros({P, 3}, vertices.options());
  torch::Tensor dL_dnormal3D = torch::zeros({P, 3}, vertices.options());

  torch::Tensor dL_dvertice_depth = torch::zeros({V}, vertices.options());

  torch::Tensor dL_dtau = torch::zeros({6}, vertices.options());
  
  if(P != 0)
  {  
	  CudaRasterizer::Rasterizer::backward(P, V, degree, M, R,
	  background.contiguous().data_ptr<float>(),
	  W, H, 
	  projmatrix_raw.contiguous().data_ptr<float>(),
	  vertices.contiguous().data_ptr<float>(),
	  triangles_indices.contiguous().data_ptr<int>(),
	  vertex_weights.contiguous().data_ptr<float>(),
	  sigma,
	  total_nb_points,
	  sh.contiguous().data_ptr<float>(),
	  colors.contiguous().data_ptr<float>(),
	  viewmatrix.contiguous().data_ptr<float>(),
	  projmatrix.contiguous().data_ptr<float>(),
	  campos.contiguous().data_ptr<float>(),
	  tan_fovx,
	  tan_fovy,
	  radii.contiguous().data_ptr<int>(),
	  reinterpret_cast<char*>(geomBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(binningBuffer.contiguous().data_ptr()),
	  reinterpret_cast<char*>(imageBuffer.contiguous().data_ptr()),
	  dL_dout_color.contiguous().data_ptr<float>(),
	  dL_dout_others.contiguous().data_ptr<float>(),
	  dL_dmeans2D.contiguous().data_ptr<float>(),
	  dL_dnormal3D.contiguous().data_ptr<float>(),
	  dL_dvertices3D.contiguous().data_ptr<float>(),
	  dL_dvertice_weight.contiguous().data_ptr<float>(),
	  dL_dnormals.contiguous().data_ptr<float>(),
	  dL_doffsets.contiguous().data_ptr<float>(),
	  dL_dopacity.contiguous().data_ptr<float>(),
	  dL_dcolors.contiguous().data_ptr<float>(),
	  dL_dsh.contiguous().data_ptr<float>(),
	  dL_dpoints2D.contiguous().data_ptr<float>(),
	  dL_dvertice_depth.contiguous().data_ptr<float>(),
	  dL_dtau.contiguous().data_ptr<float>(),
	  debug,
	  optimize_pose_only,
	  opacity_min);
  }

  return std::make_tuple(dL_dvertices3D, dL_dvertice_weight, dL_dcolors, dL_dsh, dL_dtau);
}

torch::Tensor markVisible(
		torch::Tensor& means3D,
		torch::Tensor& viewmatrix,
		torch::Tensor& projmatrix)
{ 
  const int P = means3D.size(0);
  
  torch::Tensor present = torch::full({P}, false, means3D.options().dtype(at::kBool));
 
  if(P != 0)
  {
	CudaRasterizer::Rasterizer::markVisible(P,
		means3D.contiguous().data_ptr<float>(),
		viewmatrix.contiguous().data_ptr<float>(),
		projmatrix.contiguous().data_ptr<float>(),
		present.contiguous().data_ptr<bool>());
  }
  
  return present;
}


std::tuple<torch::Tensor, torch::Tensor> ComputeRelocationCUDA(
	torch::Tensor& opacity_old,
	torch::Tensor& scale_old,
	torch::Tensor& N,
	torch::Tensor& binoms,
	const int n_max)
{
	const int P = opacity_old.size(0);
  
	torch::Tensor final_opacity = torch::full({P}, 0, opacity_old.options().dtype(torch::kFloat32));
	torch::Tensor final_scale = torch::full({3 * P}, 0, scale_old.options().dtype(torch::kFloat32));

	if(P != 0)
	{
		UTILS::ComputeRelocation(P,
			opacity_old.contiguous().data<float>(),
			scale_old.contiguous().data<float>(),
			N.contiguous().data<int>(),
			binoms.contiguous().data<float>(),
			n_max,
			final_opacity.contiguous().data<float>(),
			final_scale.contiguous().data<float>());
	}

	return std::make_tuple(final_opacity, final_scale);

}


void adamUpdate(
	torch::Tensor &param,
	torch::Tensor &param_grad,
	torch::Tensor &exp_avg,
	torch::Tensor &exp_avg_sq,
	torch::Tensor &visible,
	const float lr,
	const float b1,
	const float b2,
	const float eps,
	const uint32_t N,
	const uint32_t M
){
	ADAM::adamUpdate(
		param.contiguous().data<float>(),
		param_grad.contiguous().data<float>(),
		exp_avg.contiguous().data<float>(),
		exp_avg_sq.contiguous().data<float>(),
		visible.contiguous().data<bool>(),
		lr,
		b1,
		b2,
		eps,
		N,
		M);
}