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

// use embedded ptx string.
extern "C" const char ptxCode[];

namespace owl_samples {

  extern "C" int main(int ac, const char **av)
  {
    optix::Context::SP context = optix::Context::create();
    
    optix::Module::SP  module
      = context->createModuleFromString(ptxCode);
    
    context->setEntryPoint(0,module,"simpleRayGen");

    context->launch(0,optix::vec2i(1));
    
    return 0;
  }
  
}
