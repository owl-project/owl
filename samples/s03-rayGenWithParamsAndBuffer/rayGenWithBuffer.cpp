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

#include "optix/Optix.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#include "deviceCode.h"

// use embedded ptx string.
extern "C" const char ptxCode[];

namespace owl_samples {

  /* we'll compute an image of given size, with a gradient from top
     to bottom using the given two colors */
  optix::vec2i imageSize(800,600);
  optix::vec3f topColor(0,1,0);
  optix::vec3f bottomColor(0,0,1);
  
  extern "C" int main(int ac, const char **av)
  {
    optix::Context::SP context
      = optix::Context::create();
    optix::Buffer::SP  frameBuffer
      = context->createHostPinnedBuffer2D(OPTIX_UINT32,imageSize);
    optix::Module::SP  module
      = context->createModuleFromString(ptxCode);

    /*! generate a raygen program (program instance, to be
        exact) ....*/
    optix::RayGenProgram::SP rayGen
      = context->createRayGenProgram(module,"rayGenWithBuffer",
                                     sizeof(RayGenParams));
    /*! ... and declare the parameters this program knows about */
    rayGen->decVar3f("topColor",offsetof(RayGenParams,topColor),vec3f(1.f));
    rayGen->decVar3f("bottomColor",offsetof(RayGenParams,bottomColor),vec3f(0.f));

    /* now assign the variables */
    rayGen->decVarBuffer("frameBuffer",
                         /*ID offset     */-1,
                         /*pointer offset*/offsetof(RayGenParams,fbSize),
                         /*size offset   */offsetof(RayGenParams,fbPointer));
    
    // (*rayGen)["frameBuffer"]->set(frameBuffer);
    // (*rayGen)["topColor"]   ->set(topColor);
    // (*rayGen)["bottomColor"]->set(bottomColor);
    rayGen["frameBuffer"] = frameBuffer;
    rayGen["topColor"]    = topColor;
    rayGen["bottomColor"] = bottomColor;

    // assign to an entry point, and launch
    context->setEntryPoint(0,rayGen);
    context->launch(0,imageSize);
    const uint32_t *pixels = (const uint32_t*)buffer->map();
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    buffer->unmap();
    
    return 0;
  }
  
}
