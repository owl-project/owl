// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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
#include <owl/common/math/vec.h>

using namespace owl;

struct LaunchParams
{
  OptixTraversableHandle world;
  vec3f sunDirection;  // pointing toward sun
  vec3f sunColor;
  float brickScale;

  int clipHeight;  // in units of bricks

  // Note: these should be bound values
  bool enableClipping;
  bool enableToonOutline; 

  int frameID;
  float4   *fbAccumBuffer;
  uint32_t *fbPtr;
  vec2i  fbSize;

};

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
  /*! base color we use for each brick */
  unsigned char *colorIndexPerBrick;
  uchar4 *colorPalette;

  /*! array/buffer of vertex indices */
  vec3i *index;
  /*! array/buffer of vertex positions */
  vec3f *vertex;

  // For flat meshes with multiple bricks per mesh
  bool isFlat = false;
  int primCountPerBrick;
  
};

// A set of bricks in a 3d grid
// Each brick is stored as indices in the grid plus a color index
struct VoxGeomData {
  uchar4 *prims;  // (xi, yi, zi, ci)
  uchar4 *colorPalette;
  bool enableToonOutline;
};

// A version of VoxGeom where primitives are larger than single bricks.
// Each OptiiX prim is a block of NxNxN bricks (N==self.bricksPerBlock)
struct VoxBlockGeomData {
  uchar3 *prims;  // xyz indices of LL corner of a block containing NxNxN bricks.
  unsigned char *colorIndices; 
  uchar4 *colorPalette;
  int bricksPerBlock;  // cubed, should be in LaunchParams too

  // No toon outline for this mode
};

/* variables for the ray generation program */
struct RayGenData
{
  struct {
    vec3f pos;
    vec3f dir_00;
    vec3f dir_du;
    vec3f dir_dv;
  } camera;
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  color0;
  vec3f  color1;
};

// Utility
inline __host__ __device__ int indexOfMaxComponent(vec3f v)
{
  if (v.x > v.y) 
    return v.x > v.z ? 0 : 2;
  else
    return v.y > v.z ? 1 : 2;
}

