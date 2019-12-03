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
  // RayGenData &rgData = *(RayGenData*)owl::getProgramDataPointer();
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program (on device %i/%i)%s\n",
           GDT_TERMINAL_LIGHT_RED,
           self.deviceIndex,
           self.deviceCount,
           GDT_TERMINAL_DEFAULT);
  }
  if (pixelID.x >= self.fbSize.x) return;
  if (pixelID.y >= self.fbSize.y) return;

  const int numRayTypes = 1;
  const int rayType = 0;

  // that's the PRD:
  vec3f color;

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  
  OptixTraversableHandle handle = self.world;
  float3                 rayOrigin 
    = self.camera.pos;//make_float3(-3,-2,-4);
  float3                 rayDirection// = make_float3(3,2,4);
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);
  float                  tmin = 1e-3f;
  float                  tmax = 1e+10f;
  float                  rayTime = 0.f;
  OptixVisibilityMask    visibilityMask = (OptixVisibilityMask)-1;
  unsigned int           rayFlags = 0;
  unsigned int           SBToffset = rayType;
  unsigned int           SBTstride = numRayTypes;
  unsigned int           missSBTIndex = rayType;
  unsigned int           p0 = 0;
  unsigned int           p1 = 0;
  owl::packPointer(&color,p0,p1 );

  optixTrace(handle,
             rayOrigin,
             rayDirection,
             tmin,
             tmax,
             rayTime,
             visibilityMask,
             rayFlags,
             SBToffset,
             SBTstride,
             missSBTIndex,
             p0,
             p1 );
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TriangleGroupData &self = owl::getProgramData<TriangleGroupData>();
  
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Ng     = normalize(cross(B-A,C-A));

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(defaultRayType)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}


inline __device__ void __boundsFunc__SphereGeom(box3f &bounds,
                                     int primID,
                                     void *geomData)
{
  printf("bounds kernel for prim %i\n",primID);
  // bounds.lower = vec3f(-1.f);
  // bounds.lower = vec3f(+1.f);
}

extern "C" __global__ void SphereGeom__boundsFuncKernel__(void  *geomData,
                                                          box3f *boundsArray,
                                                          int    numPrims)
{
  int primID = threadIdx.x;
  printf("boundskernel - %i\n",primID);
  // if (primID < numPrims)
  //   __boundsFunc__SphereGeom(boundsArray[primID],primID,geomData);
}

