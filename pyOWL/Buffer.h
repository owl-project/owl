// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include "pyOWL/common.h"
#include "pyOWL/Module.h"

namespace pyOWL {

  struct Context;
  
  struct Buffer : public std::enable_shared_from_this<Buffer>
  {
    typedef std::shared_ptr<Buffer> SP;

    Buffer(OWLBuffer handle,
           OWLDataType type,
           size_t count)
      : handle(handle),
        type(type),
        count(count)
    {}

    inline OWLBuffer getHandle() const { return handle; }
    
    OWLBuffer handle = 0;
    OWLDataType type;
    size_t count;
  };

  struct HostPinnedBuffer : public Buffer
  {
    inline HostPinnedBuffer(OWLBuffer handle,
           OWLDataType type,
           size_t count) 
      : Buffer(handle,type,count)
    {}
    
    static Buffer::SP create(Context *context,
                             OWLDataType type,
                             size_t size);
  };
  
  struct DeviceBuffer : public Buffer
  {
    inline DeviceBuffer(OWLBuffer handle,
                        OWLDataType type,
                        size_t count)
      : Buffer(handle,type,count)
    {}
    template<typename T, int N>
    static Buffer::SP createFromCopyable(Context *ctx,
                                         OWLDataType type,
                                         const py::buffer &buffer);
    static Buffer::SP create(Context *ctx,
                             OWLDataType type,
                             const py::buffer &buffer);
  };
    
}
