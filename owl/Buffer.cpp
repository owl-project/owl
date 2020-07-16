// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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
                 OWLDataType type
                 // ,
                 // size_t elementCount
                 )
    : RegisteredObject(context,context->buffers),
      type(type)
      // ,
      //   elementCount(elementCount)
  {
    // assert(elementCount > 0);
  }

  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP Buffer::createOn(const DeviceContext::SP &device)
  {
    return std::make_shared<Buffer::DeviceData>(device);
  }

  Buffer::~Buffer()
  {
    destroy();
  }
  
  // const void *Buffer::getPointer(int deviceID)
  // {
  //   return context->llo->bufferGetPointer(this->ID,deviceID);
  // }

  size_t Buffer::getElementCount() const
  {
    return elementCount;
  }

  void HostPinnedBuffer::resize(size_t newElementCount)
  {
    if (cudaHostPinnedMem) {
      CUDA_CALL_NOTHROW(FreeHost(cudaHostPinnedMem));
      cudaHostPinnedMem = nullptr;
    }

    elementCount = newElementCount;
    if (newElementCount > 0)
      CUDA_CALL(MallocHost((void**)&cudaHostPinnedMem, sizeInBytes()));

    for (auto device : context->getDevices())
      getDD(device).d_pointer = cudaHostPinnedMem;
  }
  
  void HostPinnedBuffer::upload(const void *sourcePtr)
  {
    assert(cudaHostPinnedMem);
    memcpy(cudaHostPinnedMem,sourcePtr,sizeInBytes());
  }
  
  void HostPinnedBuffer::upload(const int deviceID, const void *hostPtr)
  {
    throw std::runtime_error("uploading to specific device doesn't "
                             "make sense for host pinned buffers");
  }
  
  // void Buffer::resize(size_t newSize)
  // {
  //   this->elementCount = newSize;
  //   return context->llo->bufferResize(this->ID,newSize*sizeOf(type));
  // }
  
  // void Buffer::upload(const void *hostPtr)
  // {
  //   context->llo->bufferUpload(this->ID,hostPtr);
  //   // lloBufferUpload(context->llo,this->ID,hostPtr);
  // }

  HostPinnedBuffer::HostPinnedBuffer(Context *const context,
                                     OWLDataType type// ,
                                     // size_t count
                                     )
    : Buffer(context,type// ,count
             )
  {
    // lloHostPinnedBufferCreate(context->llo,
    // context->llo->hostPinnedBufferCreate(this->ID,
    //                                      count*sizeOf(type),1);
  }
  
  ManagedMemoryBuffer::ManagedMemoryBuffer(Context *const context,
                                           OWLDataType type// ,
                                           // size_t count
                                           // ,
                                           // /*! data with which to
                                           //   populate this buffer; may
                                           //   be null, but has to be of
                                           //   size 'amount' if not */
                                           // const void *initData
                                           )
    : Buffer(context,type// ,count
             )
  {
    // lloManagedMemoryBufferCreate(context->llo,
    // context->llo->managedMemoryBufferCreate(this->ID,
    //                                         count*sizeOf(type),1,
    //                                         initData);
    // if (initData) throw std::runtime_error("not implemented");
  }

  void ManagedMemoryBuffer::resize(size_t newElementCount)
  {
    if (cudaManagedMem) {
      CUDA_CALL_NOTHROW(Free(cudaManagedMem));
      cudaManagedMem = 0;
    }
    
    elementCount = newElementCount;
    if (newElementCount > 0) {
      CUDA_CALL(MallocManaged((void**)&cudaManagedMem, sizeInBytes()));
      unsigned char *mem_end = (unsigned char *)cudaManagedMem + sizeInBytes();
      size_t pageSize = 16*1024*1024;
      int pageID = 0;
      for (unsigned char *begin = (unsigned char *)cudaManagedMem;
           begin < mem_end;
           begin += pageSize)
        {
          unsigned char *end = std::min(begin+pageSize,mem_end);
          int devID = pageID++ % context->deviceCount();
          int cudaDevID = context->getDevice(devID)->getCudaDeviceID();
          int result = 0;
          cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, cudaDevID);
          if (result) {
            CUDA_CALL(MemAdvise((void*)begin, end-begin,
                                cudaMemAdviseSetPreferredLocation, cudaDevID));
          }
        }
    }
    
    for (auto device : context->getDevices())
      getDD(device).d_pointer = cudaManagedMem;
  }
  
  void ManagedMemoryBuffer::upload(const void *hostPtr)
  {
    assert(cudaManagedMem);
    cudaMemcpy(cudaManagedMem,hostPtr,
               sizeInBytes(),cudaMemcpyDefault);
  }
  
  void ManagedMemoryBuffer::upload(const int deviceID,
                                   const void *hostPtr)
  {
    throw std::runtime_error("copying to a specific device doesn't"
                             " make sense for a managed mem buffer");
  }

  DeviceBuffer::DeviceData::~DeviceData()
  {
    if (d_pointer == 0) return;

    SetActiveGPU forLifeTime(device);
    
    CUDA_CALL_NOTHROW(Free(d_pointer));
    d_pointer = nullptr;
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP DeviceBuffer::createOn(const DeviceContext::SP &device)
  {
    if (type >= _OWL_BEGIN_COPYABLE_TYPES)
      return std::make_shared<DeviceBuffer::DeviceDataForCopyableData>(this,device);

    if (type == OWL_BUFFER)
      return std::make_shared<DeviceBuffer::DeviceDataForBuffers>(this,device);

    if (type == OWL_TEXTURE)
      return std::make_shared<DeviceBuffer::DeviceDataForTextures>(this,device);

    throw std::runtime_error("unsupported element type for device buffer");
  }
  
  void DeviceBuffer::upload(const void *hostPtr)
  {
    assert(deviceData.size() == context->deviceCount());
    for (auto dd : deviceData)
      dd->as<DeviceBuffer::DeviceData>()->uploadAsync(hostPtr);
    CUDA_SYNC_CHECK();
  }
  
  void DeviceBuffer::upload(const int deviceID, const void *hostPtr) 
  {
    // assert(hostPtr);
    // if (type >= _OWL_BEGIN_COPYABLE_TYPES) {
    //   /* these are copyable types, nothing to do on the host */
    // } else {
    //   /*! these are handles to virtual object types; create a
    //       host-copy of those types to properly do refcounting on them
    //       (in case anybody releases their handles after creating a
    //       buffer of them */
    //   hostHandles.resize(elementCount);    
    //   APIHandle **handles = (APIHandle **)hostPtr;
    //   for (int i=0;i<elementCount;i++) {
    //     APIHandle *handle = handles[i];
    //     hostHandles[i]
    //       = handle
    //       ? handle->object
    //       : nullptr;
    //   }
    // }
    assert(deviceID < deviceData.size());
    deviceData[deviceID]->as<DeviceBuffer::DeviceData>()->uploadAsync(hostPtr);
    CUDA_SYNC_CHECK();
  }
  

  DeviceBuffer::DeviceBuffer(Context *const context,
                             OWLDataType type// ,
                             // size_t count
                             // ,
                             // const void *initData
                             )
    : Buffer(context,type// ,count
             )
  {
#if 0
    if (type >= _OWL_BEGIN_COPYABLE_TYPES) {
      // resize(count);
      // if (initData) upload(initData);
      // context->llo->deviceBufferCreate(this->ID,
      //                                  count*sizeOf(type),1,
      //                                  initData);
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
#endif
  }

  void DeviceBuffer::resize(size_t newElementCount)
  {
    elementCount = newElementCount;
    for (auto device : context->getDevices()) 
      getDD(device).executeResize();
  }
  


  void DeviceBuffer::DeviceDataForTextures::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(cudaTextureObject_t)));
    
  }
  
  void DeviceBuffer::DeviceDataForTextures::uploadAsync(const void *hostDataPtr) 
  {
    SetActiveGPU forLifeTime(device);
    
    hostHandles.resize(parent->elementCount);
    APIHandle **apiHandles = (APIHandle **)hostDataPtr;
    std::vector<cudaTextureObject_t> devRep(parent->elementCount);
    
    for (int i=0;i<parent->elementCount;i++)
      if (apiHandles[i]) {
        Texture::SP texture = apiHandles[i]->object->as<Texture>();
        assert(texture && "make sure those are really textures in this buffer!");
        devRep[i] = texture->textureObjects[device->ID];
        hostHandles[i] = texture;
      } else
        hostHandles[i] = nullptr;

    CUDA_CALL(MemcpyAsync(d_pointer,devRep.data(),
                          devRep.size()*sizeof(devRep[0]),
                          cudaMemcpyDefault,
                          device->getStream()));
  }
  
  void DeviceBuffer::DeviceDataForBuffers::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(device::Buffer)));
  }
  
  void DeviceBuffer::DeviceDataForBuffers::uploadAsync(const void *hostDataPtr) 
  {
    SetActiveGPU forLifeTime(device);
    
    hostHandles.resize(parent->elementCount);
    APIHandle **apiHandles = (APIHandle **)hostDataPtr;
    std::vector<device::Buffer> devRep(parent->elementCount);
    
    for (int i=0;i<parent->elementCount;i++)
      if (apiHandles[i]) {
        Buffer::SP buffer = apiHandles[i]->object->as<Buffer>();
        assert(buffer && "make sure those are really textures in this buffer!");
        
        devRep[i].data    = (void*)buffer->getPointer(device);
        devRep[i].type    = buffer->type;
        devRep[i].count   = buffer->getElementCount();
        
        hostHandles[i] = buffer;
      } else {
        devRep[i].data    = 0;
        devRep[i].type    = OWL_INVALID_TYPE;
        devRep[i].count   = 0;
      }

    CUDA_CALL(MemcpyAsync(d_pointer,devRep.data(),
                          devRep.size()*sizeof(devRep[0]),
                          cudaMemcpyDefault,
                          device->getStream()));
  }
  
  void DeviceBuffer::DeviceDataForCopyableData::executeResize() 
  {
    SetActiveGPU forLifeTime(device);
    
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeOf(parent->type)));
  }
  
  void DeviceBuffer::DeviceDataForCopyableData::uploadAsync(const void *hostDataPtr)
  {
    SetActiveGPU forLifeTime(device);
    
    CUDA_CALL(MemcpyAsync(d_pointer,hostDataPtr,
                          parent->elementCount*sizeOf(parent->type),
                          cudaMemcpyDefault,
                          device->getStream()));
  }
  



  

  GraphicsBuffer::GraphicsBuffer(Context *const context,
                                 OWLDataType type,
                                 // size_t count,
                                 cudaGraphicsResource_t resource)
    : Buffer(context, type//, count
             )
  {
    // context->llo->graphicsBufferCreate(this->ID,
    //                                    count,sizeOf(type),
    //                                    resource);
  }

  void GraphicsBuffer::resize(size_t newElementCount)
  {
    elementCount = newElementCount;
  }


  void GraphicsBuffer::upload(const void *hostPtr)
  {
    throw std::runtime_error("Buffer::upload doesn' tmake sense for graphics buffers");
  }
  
  void GraphicsBuffer::upload(const int deviceID, const void *hostPtr) 
  {
    throw std::runtime_error("Buffer::upload doesn' tmake sense for graphics buffers");
  }
  



  void GraphicsBuffer::map(const int deviceID, CUstream stream)
  {
    DeviceContext::SP device = context->getDevice(deviceID);
    DeviceData &dd = getDD(device);
    // void GraphicsBuffer::map(Device *device, CUstream stream)
    // {
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, stream));
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&dd.d_pointer, &size, resource));
    // if (elementCount * elementSize != size)
    //   {
    //     throw std::runtime_error("mapped resource has unexpected size");
    //   }
  }

  void GraphicsBuffer::unmap(const int deviceID, CUstream stream)
  {
    DeviceContext::SP device = context->getDevice(deviceID);
    DeviceData &dd = getDD(device);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, stream));
    dd.d_pointer = nullptr;
  }

  /*! destroy whatever resouces this buffer's ll-layer handle this
    may refer to; this will not destruct the current object
    itself, but should already release all its references */
  void Buffer::destroy()
  {
    if (ID < 0)
      /* already destroyed */
      return;

    deviceData.clear();
    
    // lloBufferDestroy(context->llo,this->ID);
    registry.forget(this); // sets ID to -1
  }
  
} // ::owl
