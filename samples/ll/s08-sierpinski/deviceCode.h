// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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

#include "ll/llowl.h"
#include "../s05-rtow/Materials.h"

using namespace owl;

// struct TrianglesGeomData
// {
//   vec3f color;
//   vec3i *index;
//   vec3f *vertex;
// };
struct LambertianPyramidMesh {
  /*! for our pyramids geometry we use triangles for the geometry, so the
      materials will actually be shared among every group of 6
      triangles */
  Lambertian *perPyramidMaterial;
  /* the vertex and index arrays for the triangle mesh */
  vec3f *vertex;
  vec3i *index;
};

struct RayGenData
{
  int deviceIndex;
  int deviceCount;
  uint32_t *fbPtr;
  vec2i  fbSize;
  OptixTraversableHandle world;
  int sbtOffset;

  struct {
      vec3f origin;
      vec3f lower_left_corner;
      vec3f horizontal;
      vec3f vertical;
    } camera;
};

struct MissProgData
{
  vec3f  color0;
  vec3f  color1;
};

