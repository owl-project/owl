// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#pragma once

#include <owl/owl.h>
#include "owl/common/math/LinearSpace.h"
#include <owl/common/math/vec.h>
#include <cuda_runtime.h>

using namespace owl;

#define RADIANCE_RAY_TYPE 0
#define SHADOW_RAY_TYPE   1

#ifdef __CUDA_ARCH__
typedef owl::RayT<0,2> RadianceRay;
typedef owl::RayT<1,2> ShadowRay;
#endif

struct BasicLight
{
  vec3f pos;
  vec3f color;
};

struct Material {
  vec3f Kd, Ks, Ka;
  float phong_exp;
  vec3f reflectivity;
};

struct SpheresGeomData
{
  /*! color of the spheres */
  vec3f* tint;
  Material material;
};

/* variables for the ray generation program */
struct RayGenData
{
  uint32_t *fbPtr;
  vec2i  fbSize;
  OptixTraversableHandle world;

  struct {
    vec3f pos;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
    float aperture_radius;
    float focal_scale;
  } camera;
};

struct LaunchParams
{
  int numLights;
  BasicLight *lights;
  vec3f ambient_light_color;
  float scene_epsilon;
  OptixTraversableHandle world;
  float4   *accumBuffer;
  int       accumID;
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  bg_color;
};

