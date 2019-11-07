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

    /*! java-style pretty-printer, for debugging */
    virtual std::string toString() override
    { return "optix::Module"; }
    
  private:
    friend class Context;
    
    static Module::SP create(Context *context,
                      const std::string &ptxCode)
    {
      Module *module = new Module(context,ptxCode);
      return Module::SP(module);
    }
    
    Module(Context *context,
           const std::string &ptxCode);
    
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

    std::string const ptxCode;
    Context    *const context;

    std::vector<PerDevice::SP> perDevice;
  };
  
} // ::optix

