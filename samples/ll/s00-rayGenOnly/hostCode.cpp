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

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << GDT_TERMINAL_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "ll00-rayGenOnly.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  LOG("ll example '" << av[0] << "' starting up");

  // owl::ll::DeviceGroup *ll
  //   = owl::ll::DeviceGroup::create();
  LLOContext llo = lloContextCreate(nullptr,0);
  
  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");
  
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);
  
  lloAllocRayGens(llo,1);
  lloRayGenCreate(llo,
                  /*program ID*/0,
                  /*module:*/0,
                  "simpleRayGen",
                  sizeof(RayGenData));
  lloBuildPrograms(llo);
  lloCreatePipeline(llo);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer")
  enum { FRAME_BUFFER=0,NUM_BUFFERS };
  // lloAllocBuffers(llo,NUM_BUFFERS);
  lloAllocBuffers(llo,NUM_BUFFERS);
  // lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));
  lloHostPinnedBufferCreate(llo,
                            /* buffer ID */FRAME_BUFFER,
                            /* #bytes    */fbSize.x*fbSize.y*sizeof(uint32_t));

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  // ----------- build raygens -----------
  LOG("building raygen program");
  // lloSbtRayGensBuild
  //   ([&](uint8_t *output,
  //        int devID,
  //        int rgID) {
  //     RayGenData *rg = (RayGenData*)output;
  //     rg->deviceIndex   = devID;
  //     rg->deviceCount = lloGetDeviceCount(llo);
  //     rg->fbSize = fbSize;
  //     rg->fbPtr  = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
  //     rg->color0 = vec3f(.8f,0.f,0.f);
  //     rg->color1 = vec3f(.8f,.8f,.8f);
  //   });
  lloSbtRayGensBuild
    (llo,[&](uint8_t *output,
             int devID,
             int rgID) {
           RayGenData *rg  = (RayGenData*)output;
           rg->deviceIndex = devID;
           rg->deviceCount = lloGetDeviceCount(llo);
           rg->fbSize      = fbSize;
           rg->fbPtr       = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
           rg->color0      = vec3f(.8f,0.f,0.f);
           rg->color1      = vec3f(.8f,.8f,.8f);
         });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  // ll->launch(0,fbSize);
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing frame buffer to " << outFileName);
  // for host pinned mem it doesn't matter which device we query...
  // const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  // lloContextDestroy(llo);
  lloContextDestroy(llo);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
