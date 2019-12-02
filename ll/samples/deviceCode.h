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

#include "gdt/math/vec.h"
// the 'actual' optix
#include <optix.h>

// TODO: move this to common api
namespace optix {
  using namespace gdt;

#ifdef __CUDA_ARCH__
  // ==================================================================
  // actual device-side "API" built-ins.
  // ==================================================================

  inline __device__ vec2i getLaunchIndex()
  {
    return (vec2i)optixGetLaunchIndex();
  }

  /*! return dimensions of a 2-dimensional optix launch. For 1- or
    3-dimensional launches we'll need separate functions */
  inline __device__ vec2i getLaunchDims()
  {
    return (vec2i)optixGetLaunchIndex();
  }

  /*! return pointer to currently running program's "SBT Data" (which
      is pretty much what in owl we call the Program Data/Program
      Variables Struct. This method returns an untyped pointer, for
      automatic type conversion see the getProgramData<T> template */
  inline __device__ const void *getProgramDataPointer()
  {
    return (const void*)optixGetSbtDataPointer();
  }

  /*! convenience type-tagged version of \see
      getProgramDataPointer. Note this function does _not_ perform any
      type-checks, it's just hard-casting the SBT pointer to the
      expected type. */
  template<typename T>
  inline __device__ const T &getProgramData()
  {
    return *(const T*)getProgramDataPointer();
  }


  // ==================================================================
  // general convenience/helper functions - may move to samples
  // ==================================================================
  inline __device__ float linear_to_srgb(float x) {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
  }

  inline __device__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __device__ uint32_t make_rgba8(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __device__ uint32_t make_rgba8(const vec4f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }
#endif
}

using gdt::vec2i;
using gdt::vec3f;
using gdt::vec3i;

struct TriangleGroupData
{
  vec3f color;
};

struct RayGenData
{
  int deviceIndex;
  int deviceCount;
  vec3f *fbPtr;
  vec2i  fbSize;
  vec3f  color0;
  vec3f  color1;
  OptixTraversableHandle world;
};

struct MissProgData
{
};

