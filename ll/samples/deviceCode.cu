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

#include "deviceCode.h"
#include <optix_device.h>

#define OPTIX_RAYGEN_PROGRAM(programName) \
  extern "C" __global__ \
  void __raygen__##programName

#define OPTIX_CLOSEST_HIT_PROGRAM(programName) \
  extern "C" __global__ \
  void __closesthit__##programName

#define OPTIX_MISS_PROGRAM(programName) \
  extern "C" __global__ \
  void __miss__##programName

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  // RayGenData &rgData = *(RayGenData*)optix::getProgramDataPointer();
  const RayGenData &rg = optix::getProgramData<RayGenData>();
  const vec2i pixelID = optix::getLaunchIndex();
  if (pixelID == optix::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program (on device %i/%i)%s\n",
           GDT_TERMINAL_LIGHT_RED,
           rg.deviceIndex,
           rg.deviceCount,
           GDT_TERMINAL_DEFAULT);
  }
  if (pixelID.x >= rg.fbSize.x) return;
  if (pixelID.y >= rg.fbSize.y) return;

// static __forceinline__ __device__ void optixTrace( OptixTraversableHandle handle,
//                                                    float3                 rayOrigin,
//                                                    float3                 rayDirection,
//                                                    float                  tmin,
//                                                    float                  tmax,
//                                                    float                  rayTime,
//                                                    OptixVisibilityMask    visibilityMask,
//                                                    unsigned int           rayFlags,
//                                                    unsigned int           SBToffset,
//                                                    unsigned int           SBTstride,
//                                                    unsigned int           missSBTIndex,
//                                                    unsigned int&          p0,
//                                                    unsigned int&          p1 );

  const int numRayTypes = 1;
  const int rayType = 0;

  OptixTraversableHandle handle = rg.world;
  float3                 rayOrigin = make_float3(0,0,0);
    float3                 rayDirection = make_float3(0,1,0);
    float                  tmin = 1e-3f;
    float                  tmax = 1e+10f;
    float                  rayTime = 0.f;
    OptixVisibilityMask    visibilityMask = 0;
    unsigned int           rayFlags = 0;
    unsigned int           SBToffset = rayType;
    unsigned int           SBTstride = numRayTypes;
    unsigned int           missSBTIndex = rayType;
    unsigned int&          p0,
    unsigned int&          p1 );
float3 org = make_float3(0,0,0);
  float3 dir = make_float3(0,1,0);
  const uint32_t prd0 = 0;
  const uint32_t prd1 = 0;
  optixTrace(rg.world,
             org,dir,1e-3f,1e+10f,0.f,0x0,
             rayType,numRayTypes,rayType,prd0,prd1);
  
  const int fbOfs = pixelID.x+rg.fbSize.x*pixelID.y;
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  rg.fbPtr[fbOfs]
    = optix::make_rgba((pattern&1) ? rg.color1 : rg.color0);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  printf("hit!!!\n");
  // if (optix::getLaunchIndex() == optix::vec2i(0))
  //   printf("Hello OptiX From your First RayGen Program\n");
}

OPTIX_MISS_PROGRAM(defaultRayType)()
{
  printf("miss!!!\n");
  // if (optix::getLaunchIndex() == optix::vec2i(0))
  //   printf("Hello OptiX From your First RayGen Program\n");
}

