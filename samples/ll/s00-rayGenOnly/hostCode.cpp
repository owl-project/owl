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

// ll00-rayGenOnly: initialize and run a ray generator shader, and nothing else

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << OWL_TERMINAL_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

// When run, this program produces this PNG as output.
// In this case the correct result is a red and light gray checkerboard,
// as nothing is actually rendered
const char *outFileName = "ll00-rayGenOnly.png";
// image resolution
const vec2i fbSize(800,600);
// camera: unused in this sample, which generates no rays. TODO: delete?
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  // Note that the output window will also show comments for almost every line executed
  LOG("ll example '" << av[0] << "' starting up");

  // Initialize CUDA and OptiX 7, and create an "owl device," a context to hold the
  // ray generation shader and output buffer. No list of IDs means use all available devices.
  // You can learn more about this and (some) other methods by hovering over it. TODO: it would be nice if more methods had this hover text with them
  LLOContext llo = lloContextCreate(nullptr,0);
  
  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");
  
  // allocate one OptiX module, which will hold our compiled RayGen shader
  lloAllocModules(llo,1);
  // PTX is the intermediate code that the CUDA deviceCode.cu shader program is converted into.
  // You can see the machine-centric PTX code in cuda_compile_ptx_1_generated_deviceCode.cu.ptx_embedded.c
  // See https://devblogs.nvidia.com/how-to-get-started-with-optix-7/
  lloModuleCreate(llo,0,ptxCode);
  // This PTX intermediate code representation is then compiled into an OptiX module.
  lloBuildModules(llo);
  
  // First, allocate room for one RayGen shader
  lloAllocRayGens(llo,1);
  // create this shader and hold on to it with the "llo" context
  lloRayGenCreate(llo,
                  /*program ID*/0,
                  /*module:*/0,
                  "simpleRayGen",       // name of the shader, in deviceCode.cu
                  sizeof(RayGenData));  // RayGenData, in deviceCode.h, is the data the RayGen shader can read and modify
 
  // (re-)builds all optix programs, with current pipeline settings 
  lloBuildPrograms(llo);
  // Create the pipeline. Note that owl will (kindly) warn there is no geometry and no miss programs defined.
  lloCreatePipeline(llo);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer")
  enum { FRAME_BUFFER=0,NUM_BUFFERS };
  lloAllocBuffers(llo,NUM_BUFFERS);
  // Create a frame buffer as page-locked, aka "pinned" memory. See CUDA documentation for benefits and more info.
  lloHostPinnedBufferCreate(llo,
                            /* buffer ID */FRAME_BUFFER,
                            /* #bytes    */fbSize.x*fbSize.y*sizeof(uint32_t));

  // ##################################################################
  // build Shader Binding Table (SBT) required to trace the groups
  // ##################################################################
  // ----------- build raygens -----------
  LOG("building raygen program");
  // Build shader binding table entry: ray generation record.
  // The lambda function [&] sets up our specific ray generation data structure,
  // rather than defining a separate function to do this. See RayGenData in deviceCode.h
  lloSbtRayGensBuild
    (llo,[&](uint8_t *output,
             int devID,
             int rgID) {
           RayGenData *rg  = (RayGenData*)output;
           rg->deviceIndex = devID;                 // for multi-GPU configurations
           rg->deviceCount = lloGetDeviceCount(llo);
           rg->fbSize      = fbSize;                // frame buffer output information
           rg->fbPtr       = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
           rg->color0      = vec3f(.8f,0.f,0.f);    // our checkerboard colors, red
           rg->color1      = vec3f(.8f,.8f,.8f);    // and light gray
         });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  // Normally launching without a hit or miss shader causes OptiX to trigger warnings.
  // Owl's wrapper call here will set up fake hit and miss records into the SBT to avoid these.
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing frame buffer to " << outFileName);
  // for host pinned memory it doesn't matter which device we query,
  // since all GPUs will share the same storage area.
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  // write frame buffer results to the PNG file
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  lloContextDestroy(llo);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
