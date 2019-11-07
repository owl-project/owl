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

#include "optix/Module.h"

namespace optix {

  struct Program : public CommonBase {
    typedef std::shared_ptr<Program> SP;

    Program(Module::SP   module,
            const std::string &programName)
      : module(module),
        programName(programName)
    {}
        
    /*! java-style pretty-printer, for debugging */
    virtual std::string toString() override
    { return "optix::Program"; }
    
    Module::SP  const module;
    std::string const programName;
  };


  struct RayGenProg {
    typedef std::shared_ptr<RayGenProg> SP;
    
    RayGenProg(Context *context,
               Program::SP program);
    
    struct PerDevice {
      typedef std::shared_ptr<PerDevice> SP;

      PerDevice(Context::PerDevice::SP context,
                Module::PerDevice::SP  module,
                RayGenProg      *const self);
      
      void create();
      void destroy();

      RayGenProg         *const self;
      Context::PerDevice::SP    context;
      Module::PerDevice::SP     module;
      OptixProgramGroupOptions  pgOptions = {};
      OptixProgramGroupDesc     pgDesc;
      OptixProgramGroup         pg;
      std::mutex                mutex;
      bool                      created = false;
    };

    Context             *const context;
    Program::SP                program;
    std::vector<PerDevice::SP> perDevice;
  };
  
} // ::optix

