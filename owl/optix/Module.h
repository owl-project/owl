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

#pragma once

#include "optix/Context.h"

namespace optix {

  struct Module : public CommonBase {
    typedef std::shared_ptr<Module> SP;
    
    struct PerDevice {
      typedef std::shared_ptr<PerDevice> SP;

      PerDevice(Context::PerDevice::SP context,
                Module *self);
      
      ~PerDevice() { destroy(); }
      
      void create();
      void destroy();
      
      std::mutex  mutex;
      OptixModule optixModule;
      
      bool        created = false;
      
      Context::PerDevice::SP context;
      Module          *const self;
    };

    Module(Context::SP context,
           const std::string &ptxCode);
    
    std::string ptxCode;
    Context::WP context;
    std::vector<PerDevice::SP> perDevice;
  };
  
  struct Program : public CommonBase
  {
    typedef std::shared_ptr<Program> SP;
    
    Module::SP  module;
    std::string programName;
  };
    
} // ::optix

