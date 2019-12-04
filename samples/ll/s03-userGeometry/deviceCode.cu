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

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  vec3f color;
  owl::trace(/*accel to trace against*/self.world,
             /*the ray to trace*/ ray,
             /*numRayTypes*/1,
             /*prd*/color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

OPTIX_INTERSECT_PROGRAM(Sphere)()
{
  const SphereGeomData &self = owl::getProgramData<SphereGeomData>();
  
  // get index of primitive we are to intersect (for this example,
  // this will always be 0, because we have N differnent *geoms* with
  // one prim each.
  int primID = optixGetPrimitiveIndex();
  printf("isec program %f %f %f ...\n",
         self.center.x,
         self.center.y,
         self.center.z);
}

OPTIX_CLOSEST_HIT_PROGRAM(Sphere)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const SphereGeomData &self = owl::getProgramData<SphereGeomData>();
  
  const vec3f org = optixGetWorldRayOrigin();
  const vec3f dir = optixGetWorldRayDirection();
  
  // compute normal:
  const vec3f Ng     = vec3f(1,1,1);//normalize(cross(B-A,C-A));

  prd = (.2f + .8f*fabs(dot(dir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}



/* defines the wrapper stuff to actually launch all the bounds
   programs from the host - todo: move to deviceAPI.h once working */
#define OPTIX_BOUNDS_PROGRAM(progName)                                  \
  /* fwd decl for the kernel func to call */                            \
  inline __device__ void __boundsFunc__##progName(void *geomData,       \
                                                  box3f &bounds,        \
                                                  int primID);          \
                                                                        \
  /* the '__global__' kernel we can get a function handle on */         \
  extern "C" __global__                                                 \
  void __boundsFuncKernel__##progName(void  *geomData,                  \
                                      box3f *boundsArray,               \
                                      int    numPrims)                  \
  {                                                                     \
    int primID = threadIdx.x;                                           \
    if (primID < numPrims) {                                            \
      printf("boundskernel - %i\n",primID);                             \
      __boundsFunc__##progName(geomData,boundsArray[primID],primID);    \
    }                                                                   \
  }                                                                     \
                                                                        \
  /* now the actual device code that the user is writing: */            \
  inline __device__ void __boundsFunc__##progName                       \
  /* program args and body supplied by user ... */
  
  
OPTIX_BOUNDS_PROGRAM(Sphere)(void  *geomData,
                             box3f &primBounds,
                             int    primID)
{
  printf("sphere bounds kernel for prim %i\n",primID);
}

