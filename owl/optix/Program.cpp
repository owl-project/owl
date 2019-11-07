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

#include "optix/Context.h"
#include "optix/Module.h"
#include "optix/Program.h"

namespace optix {

  RayGenProg::RayGenProg(Context *context,
                         Program::SP program)
    : context(context),
      program(program)
  {
    assert(context);
    std::lock_guard<std::mutex> lock(context->mutex);
    for (size_t i=0;i<context->perDevice.size();i++)
      perDevice.push_back
        (std::make_shared<PerDevice>(context->perDevice[i],this));
  }

  RayGenProg::PerDevice::PerDevice(Context::PerDevice::SP context,
                                   RayGenProg           *const self)
    : context(context),
      self(self)
  {}
  
}
