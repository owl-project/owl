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
#include "APIHandle.h"
#include "owl/owl_device_buffer.h"

namespace owl {

  Buffer::Buffer(Context *const context,
                 OWLDataType type,
                 size_t dims)
    : RegisteredObject(context,context->buffers),
      type(type),
      elementCount(elementCount)
  {}

  Buffer::~Buffer()
  {
    destroy();
  }
  
  const void *Buffer::getPointer(int deviceID)
  {
    return context->llo->bufferGetPointer(this->ID,deviceID);
  }

  size_t Buffer::getElementCount() const
  {
    return elementCount;
  }

  void Buffer::resize(size_t newSize)
  {
    this->elementCount = newSize;
    return context->llo->bufferResize(this->ID,newSize*sizeOf(type));
  }
  
  void Buffer::upload(const void *hostPtr)
  {
    context->llo->bufferUpload(this->ID,hostPtr);
    // lloBufferUpload(context->llo,this->ID,hostPtr);
  }

  HostPinnedBuffer::HostPinnedBuffer(Context *const context,
                                     OWLDataType type,
                                     size_t count)
    : Buffer(context,type,count)
  {
    // lloHostPinnedBufferCreate(context->llo,
    context->llo->hostPinnedBufferCreate(this->ID,
                                         count*sizeOf(type),1);
  }
  
  ManagedMemoryBuffer::ManagedMemoryBuffer(Context *const context,
                                           OWLDataType type,
                                           size_t count,
                                           /*! data with which to
                                             populate this buffer; may
                                             be null, but has to be of
                                             size 'amount' if not */
                                           const void *initData)
    : Buffer(context,type,count)
  {
    // lloManagedMemoryBufferCreate(context->llo,
    context->llo->managedMemoryBufferCreate(this->ID,
                                            count*sizeOf(type),1,
                                            initData);
  }
  
  DeviceBuffer::DeviceBuffer(Context *const context,
                             OWLDataType type,
                             size_t count,
                             const void *initData)
    : Buffer(context,type,count)
  {
    if (type < _OWL_BEGIN_NON_COPYABLE_TYPES) {
      context->llo->deviceBufferCreate(this->ID,
                                       count*sizeOf(type),1,
                                       initData);
    } else if (type == OWL_BUFFER) {
      if (!initData)
        throw std::runtime_error("buffers with type OWL_BUFFER _have_ to specify the buffer handles at creation time");
      for (int i=0;i<count;i++) {
        APIHandle *handle = ((APIHandle**)initData)[i];
        Buffer::SP buffer;
        if (handle) {
          buffer = handle->get<Buffer>();
          if (!buffer)
            throw std::runtime_error
              ("trying to create a buffer of buffers, but at least "
               "one handle in the init memory is not a buffer");
        }
        hostHandles.push_back(buffer);
      }
      // create the buffer, with empty values.
      std::vector<device::Buffer> devRep(count);
      context->llo->deviceBufferCreate(this->ID,
                                       devRep.size()*sizeof(devRep[0]),1,
                                       nullptr);
      
      // now, set the device-specific values
      for (int devID = 0; devID < context->llo->devices.size(); devID++) {
        for (int i=0;i<count;i++)
          if (hostHandles[i]) {
            Buffer::SP buffer = hostHandles[i]->as<Buffer>();
            
            devRep[i].data    = (void*)buffer->getPointer(devID);
            devRep[i].type    = buffer->type;
            devRep[i].count   = buffer->getElementCount();
          } else {
            devRep[i].data    = 0;
            devRep[i].type    = OWL_INVALID_TYPE;
            devRep[i].count   = 0;
          }
        context->llo->bufferUploadToSpecificDevice(this->ID,devID,devRep.data());
      }
    } else if (type == OWL_TEXTURE) {
      if (!initData)
        throw std::runtime_error("buffers with type OWL_TEXTURE _have_ to specify the texture handles at creation time");
      for (int i=0;i<count;i++) {
        APIHandle *handle = ((APIHandle**)initData)[i];
        Texture::SP texture;
        if (handle) {
          texture = handle->get<Texture>();
          if (!texture)
            throw std::runtime_error
              ("trying to create a buffer of textures, but at least "
               "one handle in the init memory is not a texture");
        }
        hostHandles.push_back(texture);
      }
      // create the buffer, with empty values.
      std::vector<cudaTextureObject_t> devRep(count);
      context->llo->deviceBufferCreate(this->ID,
                                       devRep.size()*sizeof(devRep[0]),1,
                                       nullptr);
      
      // now, set the device-specific values
      for (int devID = 0; devID < context->llo->devices.size(); devID++) {
        for (int i=0;i<count;i++)
          if (hostHandles[i]) {
            Texture::SP texture = hostHandles[i]->as<Texture>();
            devRep[i] = texture->textureObjects[devID];
          }
        context->llo->bufferUploadToSpecificDevice(this->ID,devID,devRep.data());
      }
    } else
      throw std::runtime_error("invalid element type in buffer creation");
  }

  GraphicsBuffer::GraphicsBuffer(Context *const context,
                                 OWLDataType type,
                                 size_t count,
                                 cudaGraphicsResource_t resource)
    : Buffer(context, type, count)
  {
    context->llo->graphicsBufferCreate(this->ID,
                                       count,sizeOf(type),
                                       resource);
  }
  
  void GraphicsBuffer::map()
  {
    context->llo->graphicsBufferMap(this->ID);
  }

  void GraphicsBuffer::unmap()
  {
    context->llo->graphicsBufferUnmap(this->ID);
  }

  /*! destroy whatever resouces this buffer's ll-layer handle this
    may refer to; this will not destruct the current object
    itself, but should already release all its references */
  void Buffer::destroy()
  {
    if (ID < 0)
      /* already destroyed */
      return;
    
    context->llo->bufferDestroy(this->ID);
    // lloBufferDestroy(context->llo,this->ID);
    registry.forget(this); // sets ID to -1
  }
  
} // ::owl
