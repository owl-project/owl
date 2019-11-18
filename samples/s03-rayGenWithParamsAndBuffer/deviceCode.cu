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

#include "optix/device.h"
#include "deviceCode.h"

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const optix::vec2i pixelID = optix::getLaunchIndex();
  const optix::vec2i fbSize  = optix::getLaunchDims();
  if (optix::any_greater_or_equal(pixelID,fbSize))
    return;
  const RayGenParams *const self
    = optix::getProgramData<RayGenParams>();
  const float blend
    = pixelID.y / float(fbSize.y-1.f);

  const optix::vec3f color
    = (1.f-blend)*self->topColor + blend*self->bottomColor;
  const int pixelIndex
    = pixelID.x+fbSize.x*pixelID.y;
  self->fbPointer[pixelIndex]
    = optix::make_rgba8(color);
}

