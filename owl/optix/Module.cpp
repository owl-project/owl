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

namespace optix {

  Module::Module(Context::SP context,
                 const std::string &ptxCode)
  {
    for (int i=0;i<context->perDevice.size();i++)
      perDevice.push_back
        (std::make_shared<PerDevice>(context->perDevice[i],this));
  }

  Module::PerDevice::PerDevice(Context::PerDevice::SP context,
                               Module *self)
    : context(context),
      self(self)
  {}

  void Module::PerDevice::create()
  {
    destroy();
        
    char log[2048];
    size_t sizeof_log = sizeof( log );
    OPTIX_CHECK(optixModuleCreateFromPTX(context->optixContext,
                                         &context->self->moduleCompileOptions,
                                         &context->self->pipelineCompileOptions,
                                         self->ptxCode.c_str(),
                                         self->ptxCode.size(),
                                         log,
                                         &sizeof_log,
                                         &optixModule
                                         ));
    if (sizeof_log > 0) PRINT(sizeof_log);
  }

  void Module::PerDevice::destroy()
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (created) {
      optixModuleDestroy(optixModule);
      created = false;
    }
  }
    
} // ::optix

