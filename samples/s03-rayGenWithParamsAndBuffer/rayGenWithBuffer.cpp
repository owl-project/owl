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
#include 

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
    
#if 0
    /* first draft - "direct" write with offsets, no names.

       Pro: 

       a) shorter. 

       b) no string matching - faster?

       Cons: 

       a) no type checking - more error prone? 

       b) will *still* need to do late binding (buffers get written
       when SBT is built, not when this function is called!), so this
       behaves differently from what it implies.

       c) late binding means we still have to track who writes what,
       where, and when. ie, still have to store a map of wihch
       'variables' get written at which offset, which more easily
       allows to shoot oneself in the foot if writes overlap (eg,
       writing to fbSize.x *and* fbSize
    */
    optix::RayGenProgram::SP rayGen
      = context->createRayGenProgram(module,"rayGenWithBuffer",
                                     sizeof(RayGenParams));
    rayGen->setVar2i(offsetof(RayGenParams,fbSize),imageSize);
    rayGen->setVar3f(offsetof(RayGenParams,topColor),topColor);
    rayGen->setVar3f(offsetof(RayGenParams,bottomColor),bottomColor);
    rayGen->setVarBufferData(offsetof(RayGenParams,fbPointer),frameBuffer);
    rayGen->setVarBufferSize(offsetof(RayGenParams,fbSize),frameBuffer);
#else
    /* current draft - go through named variables that have to be exported

       con: 

       a) need string matching to find variables. cost???

       b) need explicit 'decalre' and 'set' phases; kind-of need
       difference between "type" and "instance" of a program (type to
       declare, instance to set)

       pro:
       
       a) more optix6-like

       b) allow type-checking when setting

       c) more obvious that variables get 'cached' 'til later? ie, not
       as easily confused with a low-level memcpy as the first
       variant?

       d) setting phase _somewhat_ easier to read than offsetof's
    */
    optix::RayGenProgram::SP rayGen
      = context->createRayGenProgram(module,"rayGenWithBuffer",
                                     sizeof(RayGenParams));
    rayGen->decVar3f("topColor",offsetof(RayGenParams,topColor),vec3f(1.f));
    rayGen->decVar3f("bottomColor",offsetof(RayGenParams,bottomColor),vec3f(0.f));
    rayGen->decVarBuffer("frameBuffer",
                         /*ID offset     */-1,
                         /*pointer offset*/offsetof(RayGenParams,fbSize),
                         /*size offset   */offsetof(RayGenParams,fbPointer));
    
    (*rayGen)["frameBuffer"]->set(frameBuffer);
    (*rayGen)["topColor"]   ->set(topColor);
    (*rayGen)["bottomColor"]->set(bottomColor);
#endif
    
    context->setEntryPoint(0,rayGen);
    context->launch(0,imageSize);
    const uint32_t *pixels = (const uint32_t*)buffer->map();
    stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                   pixels.data(),fbSize.x*sizeof(uint32_t));
    buffer->unmap();
    
    return 0;
  }
  
}
