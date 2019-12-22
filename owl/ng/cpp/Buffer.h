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

#include "RegisteredObject.h"

namespace owl {

  struct Buffer : public RegisteredObject
  {
    typedef std::shared_ptr<Buffer> SP;
    
    Buffer(Context *const context);
    
    virtual std::string toString() const { return "Buffer"; }

    const void *getPointer(int deviceID);
  };

  struct HostPinnedBuffer : public Buffer {
    typedef std::shared_ptr<HostPinnedBuffer> SP;
    
    HostPinnedBuffer(Context *const context) : Buffer(context) {}
    
    virtual std::string toString() const { return "HostPinnedBuffer"; }
  };
  
  struct DeviceBuffer : public Buffer {
    typedef std::shared_ptr<DeviceBuffer> SP;
    
    DeviceBuffer(Context *const context) : Buffer(context) {}
    
    virtual std::string toString() const { return "DeviceBuffer"; }
  };
  
} // ::owl
