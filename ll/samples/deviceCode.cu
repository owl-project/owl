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
  
  const int fbOfs = pixelID.x+rg.fbSize.x*pixelID.y;
  rg.fbPtr[fbOfs]
    = (fbOfs%2)
    ? rg.color1
    : rg.color0;
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  // if (optix::getLaunchIndex() == optix::vec2i(0))
  //   printf("Hello OptiX From your First RayGen Program\n");
}

OPTIX_MISS_PROGRAM(defaultRayType)()
{
  // if (optix::getLaunchIndex() == optix::vec2i(0))
  //   printf("Hello OptiX From your First RayGen Program\n");
}

