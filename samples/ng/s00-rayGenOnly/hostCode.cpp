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

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << OWL_TERMINAL_BLUE;                       \
  std::cout << "#owl.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#owl.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "s00-rayGenOnly.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  LOG("owl example '" << av[0] << "' starting up");

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");

  OWLContext owl
    = owlContextCreate();
  OWLModule module
    = owlModuleCreate(owl,ptxCode);

  OWLVarDecl rayGenVars[]
    = {
       { "fbPtr",  OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr) },
       { "fbSize", OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize) },
       { "color0", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,color0) },
       { "color1", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,color1) },
       { /* sentinel: */ nullptr }
  };
  OWLRayGen rayGen
    = owlRayGenCreate(owl,module,"simpleRayGen",
                      sizeof(RayGenData),rayGenVars,-1);
  owlBuildPrograms(owl);
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer");
  OWLBuffer
    frameBuffer = owlHostPinnedBufferCreate(owl,
                                            /*type:*/OWL_INT,
                                            /*size:*/fbSize.x*fbSize.y);

  // ------------------------------------------------------------------
  // build *SBT* required to trace the groups
  // ------------------------------------------------------------------

  owlRayGenSet3f(rayGen,"color0",.8f,0.f,0.f);
  owlRayGenSet3f(rayGen,"color1",.8f,.8f,.8f);
  owlRayGenSetBuffer(rayGen,"fbPtr",frameBuffer);
  owlRayGenSet2i(rayGen,"fbSize",fbSize.x,fbSize.y);
  owlBuildSBT(owl);

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing frame buffer to " << outFileName);
  const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("cleaning up ...");
  owlContextDestroy(owl);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
