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

  // ------------------------------------------------------------------
  // Buffer::DeviceData
  // ------------------------------------------------------------------
  
  /*! constructor */
  Buffer::DeviceData::DeviceData(const DeviceContext::SP &device)
    : RegisteredObject::DeviceData(device)
  {}

  // ------------------------------------------------------------------
  // Buffer
  // ------------------------------------------------------------------
  
  Buffer::Buffer(Context *const context,
                 OWLDataType type)
    : RegisteredObject(context,context->buffers),
      type(type)
  {
  }

  Buffer::~Buffer()
  {
    destroy();
  }
  
  /*! pretty-printer, for printf-debugging */
  std::string Buffer::toString() const 
  {
    return "Buffer";
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
    
    registry.forget(this); // sets ID to -1
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP Buffer::createOn(const DeviceContext::SP &device)
  {
    return std::make_shared<Buffer::DeviceData>(device);
  }

  // ------------------------------------------------------------------
  // Device Buffer
  // ------------------------------------------------------------------
  
  /*! any device-specific data, such as optix handles, cuda device
    pointers, etc */
  DeviceBuffer::DeviceData::DeviceData(DeviceBuffer *parent, const DeviceContext::SP &device)
    : Buffer::DeviceData(device), parent(parent)
  {}

  /*! pretty-printer, for debugging */
  std::string DeviceBuffer::toString() const 
  {
    return "DeviceBuffer";
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
      dd->as<DeviceBuffer::DeviceData>().uploadAsync(hostPtr);
    CUDA_SYNC_CHECK();
  }
  
  void DeviceBuffer::upload(const int deviceID, const void *hostPtr) 
  {
    assert(deviceID < deviceData.size());
    deviceData[deviceID]->as<DeviceBuffer::DeviceData>().uploadAsync(hostPtr);
    CUDA_SYNC_CHECK();
  }
  

  DeviceBuffer::DeviceBuffer(Context *const context,
                             OWLDataType type)
    : Buffer(context,type)
  {}

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
  

  

  // ------------------------------------------------------------------
  // Host Pinned Buffer
  // ------------------------------------------------------------------
  
  HostPinnedBuffer::HostPinnedBuffer(Context *const context,
                                     OWLDataType type)
    : Buffer(context,type)
  {
  }
  
  /*! pretty-printer, for debugging */
  std::string HostPinnedBuffer::toString() const
  {
    return "HostPinnedBuffer";
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

    for (auto device : context->getDevices()) {
      getDD(device).d_pointer = cudaHostPinnedMem;
    }
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
  
  // ------------------------------------------------------------------
  // Managed Mem Buffer
  // ------------------------------------------------------------------
  
  ManagedMemoryBuffer::ManagedMemoryBuffer(Context *const context,
                                           OWLDataType type)
    : Buffer(context,type)
  {}

  /*! pretty-printer, for debugging */
  std::string ManagedMemoryBuffer::toString() const
  {
    return "ManagedMemoryBuffer";
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

  // ------------------------------------------------------------------
  // Graphics Resource Buffer
  // ------------------------------------------------------------------
  
  /*! pretty-printer, for debugging */
  std::string GraphicsBuffer::toString() const
  {
    return "GraphicsBuffer";
  }

  GraphicsBuffer::GraphicsBuffer(Context *const context,
                                 OWLDataType type,
                                 cudaGraphicsResource_t resource)
    : Buffer(context, type)
  {}

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
    CUDA_CHECK(cudaGraphicsMapResources(1, &resource, stream));
    size_t size = 0;
    CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&dd.d_pointer, &size, resource));
  }

  void GraphicsBuffer::unmap(const int deviceID, CUstream stream)
  {
    DeviceContext::SP device = context->getDevice(deviceID);
    DeviceData &dd = getDD(device);
    CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, stream));
    dd.d_pointer = nullptr;
  }

} // ::owl
