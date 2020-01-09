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

#include "Buffer.h"
#include "Context.h"

namespace owl {

  Buffer::Buffer(Context *const context,
                 OWLDataType type)
    : RegisteredObject(context,context->buffers),
      type(type)
  {}

  const void *Buffer::getPointer(int deviceID)
  {
    return lloBufferGetPointer(context->llo,this->ID,deviceID);
  }

  void Buffer::resize(size_t newSize)
  {
    lloBufferResize(context->llo,this->ID,newSize*sizeOf(type));
  }
  
  void Buffer::upload(const void *hostPtr)
  {
    lloBufferUpload(context->llo,this->ID,hostPtr);
  }

  HostPinnedBuffer::HostPinnedBuffer(Context *const context,
                                     OWLDataType type,
                                     size_t count)
    : Buffer(context,type)
  {
    lloHostPinnedBufferCreate(context->llo,
                              this->ID,
                              count*sizeOf(type));
  }
  
  DeviceBuffer::DeviceBuffer(Context *const context,
                             OWLDataType type,
                             size_t count,
                             const void *init)
    : Buffer(context,type)
  {
    lloDeviceBufferCreate(context->llo,
                          this->ID,
                          count*sizeOf(type),
                          init);
  }


  /*! destroy whatever resouces this buffer's ll-layer handle this
    may refer to; this will not destruct the current object
    itself, but should already release all its references */
  void Buffer::destroy()
  {
    if (ID < 0)
      /* already destroyed */
      return;
    
    lloBufferDestroy(context->llo,this->ID);
    registry.forget(this); // sets ID to -1
  }
  
} // ::owl
