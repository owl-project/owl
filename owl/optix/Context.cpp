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

namespace optix {

  void Context::log_cb(unsigned int level,
                       const char *tag,
                       const char *message,
                       void * /*cbdata */)
  {
    fprintf( stderr, "#owl: [%2d][%12s]: %s\n", level, tag, message );
  }
  

    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown. 

        will throw an error if no device(s) could be found for this context

        Should never be called directly, only through Context::create() */
  Context::Context(const std::vector<uint32_t> &deviceIDs)
  {
    Device::g_init();
    
    std::cout << "#owl.context: creating new owl context..." << std::endl;
    for (auto ID : deviceIDs) {
      Device::SP device = Device::getDevice(ID);
      if (device) 
        devices.push_back(device);
    }
    if (devices.empty())
      throw optix::Error("Context::create(deviceIDs)",
                         "could not create any devices"
                         );
  }

    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown */
  Context::SP Context::create(const std::vector<uint32_t> &deviceIDs)
  {
    return std::make_shared<Context>(deviceIDs);
  }

  /*! creates a new context with the given device IDs. Invalid
    device IDs get ignored with a warning, but if no device can be
    created at all an error will be thrown */
  Context::SP Context::create(GPUSelectionMethod whichGPUs)
  {
    switch(whichGPUs) {
    case Context::GPU_SELECT_FIRST:
      return create((std::vector<uint32_t>){0});
    default:
      throw optix::Error("Context::create(GPUSelectionMethod)",
                         "specified GPU Selection method not implemented "
                         "(method="+std::to_string((int)whichGPUs)+")"
                         );
    }
  }
  
} // ::optix
