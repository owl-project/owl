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
  Buffer::DeviceData::SP Buffer::createOn(ll::Device *device)
  {
    return std::make_shared<Buffer::DeviceData>();
  }

  void Buffer::createDeviceData(const std::vector<ll::Device *> &devices)
  {
    for (auto device : devices)
      deviceData.push_back(createOn(device));
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
    
    for (auto dd : deviceData)
      dd->d_pointer = cudaHostPinnedMem;
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
          int devID = pageID++ % context->llo->devices.size();
          int cudaDevID = context->llo->devices[devID]->getCudaDeviceID();
          int result = 0;
          cudaDeviceGetAttribute(&result, cudaDevAttrConcurrentManagedAccess, cudaDevID);
          if (result) {
            CUDA_CALL(MemAdvise((void*)begin, end-begin,
                                cudaMemAdviseSetPreferredLocation, cudaDevID));
          }
        }
    }
    
    for (auto dd : deviceData)
      dd->d_pointer = cudaManagedMem;
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

    int oldActive = device->pushActive();
    CUDA_CALL_NOTHROW(Free(d_pointer));
    d_pointer = nullptr;
    device->popActive(oldActive);
  }

  /*! creates the device-specific data for this group */
  Buffer::DeviceData::SP DeviceBuffer::createOn(ll::Device *device)
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
    assert(deviceData.size() == context->llo->devices.size());
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
    for (auto device : context->llo->devices) 
      deviceData[device->ID]->as<DeviceBuffer::DeviceData>()->executeResize();
  }
  


  void DeviceBuffer::DeviceDataForTextures::executeResize() 
  {
    int oldActive = device->pushActive();
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(cudaTextureObject_t)));
    
    device->popActive(oldActive);
  }
  
  void DeviceBuffer::DeviceDataForTextures::uploadAsync(const void *hostDataPtr) 
  {
    int oldActive = device->pushActive();
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
    device->popActive(oldActive);
  }
  
  void DeviceBuffer::DeviceDataForBuffers::executeResize() 
  {
    int oldActive = device->pushActive();
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeof(device::Buffer)));
    
    device->popActive(oldActive);
  }
  
  void DeviceBuffer::DeviceDataForBuffers::uploadAsync(const void *hostDataPtr) 
  {
    int oldActive = device->pushActive();
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
    device->popActive(oldActive);
  }
  
  void DeviceBuffer::DeviceDataForCopyableData::executeResize() 
  {
    int oldActive = device->pushActive();
    if (d_pointer) { CUDA_CALL(Free(d_pointer)); d_pointer = nullptr; }

    if (parent->elementCount)
      CUDA_CALL(Malloc(&d_pointer,parent->elementCount*sizeOf(parent->type)));
    
    device->popActive(oldActive);
  }
  
  void DeviceBuffer::DeviceDataForCopyableData::uploadAsync(const void *hostDataPtr)
  {
    int oldActive = device->pushActive();
    CUDA_CALL(MemcpyAsync(d_pointer,hostDataPtr,
                          parent->elementCount*sizeOf(parent->type),
                          cudaMemcpyDefault,
                          device->getStream()));
    device->popActive(oldActive);
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
  



  void GraphicsBuffer::map(int deviceID, CUstream stream)
  {
    DeviceData &dd = *deviceData[deviceID];
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

  void GraphicsBuffer::unmap(int deviceID, CUstream stream)
  {
    DeviceData &dd = *deviceData[deviceID];
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




#if 0
void Device::bufferDestroy(int bufferID)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID <  buffers.size());
  assert("check buffer to be destroyed actually exists"
         && buffers[bufferID] != nullptr);
  int oldActive = context->pushActive();
  delete buffers[bufferID];
  buffers[bufferID] = nullptr;
  context->popActive(oldActive);
}
    
void Device::deviceBufferCreate(int bufferID,
                                size_t elementCount,
                                size_t elementSize,
                                const void *initData)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID <  buffers.size());
  assert("check buffer ID available" && buffers[bufferID] == nullptr);
  STACK_PUSH_ACTIVE(context);
  // context->pushActive();
  DeviceBuffer *buffer = new DeviceBuffer(elementCount,elementSize);
  if (initData) {
    buffer->devMem.upload(initData,"createDeviceBuffer: uploading initData");
    // LOG("uploading " << elementCount
    //     << " items of size " << elementSize
    //     << " from host ptr " << initData
    //     << " to device ptr " << buffer->devMem.get());
  }
  assert("check buffer properly created" && buffer != nullptr);
  buffers[bufferID] = buffer;
  // context->popActive();
  STACK_POP_ACTIVE();
}
    
/*! create a managed memory buffer */
void Device::managedMemoryBufferCreate(int bufferID,
                                       size_t elementCount,
                                       size_t elementSize,
                                       ManagedMemory::SP managedMem)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID <  buffers.size());
  assert("check buffer ID available" && buffers[bufferID] == nullptr);
  int oldActive = context->pushActive();
  Buffer *buffer = new ManagedMemoryBuffer(elementCount,elementSize,managedMem);
  assert("check buffer properly created" && buffer != nullptr);
  buffers[bufferID] = buffer;
  context->popActive(oldActive);
}
      
void Device::hostPinnedBufferCreate(int bufferID,
                                    size_t elementCount,
                                    size_t elementSize,
                                    HostPinnedMemory::SP pinnedMem)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID <  buffers.size());
  assert("check buffer ID available" && buffers[bufferID] == nullptr);
  int oldActive = context->pushActive();
  Buffer *buffer = new HostPinnedBuffer(elementCount,elementSize,pinnedMem);
  assert("check buffer properly created" && buffer != nullptr);
  buffers[bufferID] = buffer;
  context->popActive(oldActive);
}

void Device::graphicsBufferCreate(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize,
                                  cudaGraphicsResource_t resource)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID < buffers.size());
  assert("check buffer ID available" && buffers[bufferID] == nullptr);
  int oldActive = context->pushActive();
  Buffer *buffer = new GraphicsBuffer(elementCount, elementSize, resource);
  assert("check buffer properly created" && buffer != nullptr);
  buffers[bufferID] = buffer;
  context->popActive(oldActive);
}

void Device::graphicsBufferMap(int bufferID)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID < buffers.size());
  int oldActive = context->pushActive();
  GraphicsBuffer *buffer = dynamic_cast<GraphicsBuffer*>(buffers[bufferID]);
  assert("check buffer properly casted" && buffer != nullptr);
  buffer->map(this, context->stream);
  context->popActive(oldActive);
}

void Device::graphicsBufferUnmap(int bufferID)
{
  assert("check valid buffer ID" && bufferID >= 0);
  assert("check valid buffer ID" && bufferID < buffers.size());
  int oldActive = context->pushActive();
  GraphicsBuffer *buffer = dynamic_cast<GraphicsBuffer*>(buffers[bufferID]);
  assert("check buffer properly casted" && buffer != nullptr);
  buffer->unmap(this, context->stream);
  context->popActive(oldActive);
}
    
      
/*! returns the given buffers device pointer */
void *Device::bufferGetPointer(int bufferID)
{
  return (void*)checkGetBuffer(bufferID)->d_pointer;
}

void Device::bufferResize(int bufferID, size_t newItemCount)
{
  checkGetBuffer(bufferID)->resize(this,newItemCount);
}
    
void Device::bufferUpload(int bufferID, const void *hostPtr)
{
  int oldActive = context->pushActive();
  checkGetBuffer(bufferID)->upload(this,hostPtr);
  context->popActive(oldActive);
}

    

// ##################################################################
// ManagedMemoryMemory
// ##################################################################

ManagedMemory::ManagedMemory(DeviceGroup *devGroup,
                             size_t amount,
                             /*! data with which to populate this buffer; may
                               be null, but has to be of size 'amount' if
                               not */
                             const void *initData)
  : devGroup(devGroup)
{
  alloc(amount);
  if (initData)
    CUDA_CALL(Memcpy(pointer,initData,amount,
                     cudaMemcpyDefault));
  assert(pointer != nullptr);
}
    
ManagedMemory::~ManagedMemory()
{
  assert(pointer != nullptr);
  free();
}

void ManagedMemory::alloc(size_t amount)
{
  CUDA_CALL(MallocManaged((void**)&pointer, amount));
  // CUDA_CALL(MemAdvise((void*)pointer, amount, cudaMemAdviseSetReadMostly, -1));
  unsigned char *mem_end = (unsigned char *)pointer + amount;
  size_t pageSize = 16*1024*1024;
  int pageID = 0;
  for (unsigned char *begin = (unsigned char *)pointer; begin < mem_end; begin += pageSize) {
    unsigned char *end = std::min(begin+pageSize,mem_end);
    int devID = pageID++ % devGroup->devices.size();
    int cudaDevID = devGroup->devices[devID]->getCudaDeviceID();
    int result = 0;
    cudaDeviceGetAttribute (&result, cudaDevAttrConcurrentManagedAccess, cudaDevID);
    if (result) {
      CUDA_CALL(MemAdvise((void*)begin, end-begin, cudaMemAdviseSetPreferredLocation, cudaDevID));
    }
  }
}
    
void ManagedMemory::free()
{
  CUDA_CALL_NOTHROW(Free(pointer));
  pointer = nullptr;
}



void DeviceGroup::bufferDestroy(int bufferID)
{
  for (auto device : devices) 
    device->bufferDestroy(bufferID);
}

void DeviceGroup::deviceBufferCreate(int bufferID,
                                     size_t elementCount,
                                     size_t elementSize,
                                     const void *initData)
{
  for (auto device : devices) 
    device->deviceBufferCreate(bufferID,elementCount,elementSize,initData);
}

void DeviceGroup::hostPinnedBufferCreate(int bufferID,
                                         size_t elementCount,
                                         size_t elementSize)
{
  HostPinnedMemory::SP pinned
    = std::make_shared<HostPinnedMemory>(elementCount*elementSize);
  for (auto device : devices) 
    device->hostPinnedBufferCreate(bufferID,elementCount,elementSize,pinned);
}

void DeviceGroup::managedMemoryBufferCreate(int bufferID,
                                            size_t elementCount,
                                            size_t elementSize,
                                            const void *initData)
{
  ManagedMemory::SP mem
    = std::make_shared<ManagedMemory>(this,elementCount*elementSize,initData);
  for (auto device : devices) 
    device->managedMemoryBufferCreate(bufferID,elementCount,elementSize,mem);
}

void DeviceGroup::graphicsBufferCreate(int bufferID,
                                       size_t elementCount,
                                       size_t elementSize,
                                       cudaGraphicsResource_t resource)
{
  for (auto device : devices)
    device->graphicsBufferCreate(bufferID, elementCount, elementSize, resource);
}

void DeviceGroup::graphicsBufferMap(int bufferID)
{
  for (auto device : devices)
    device->graphicsBufferMap(bufferID);
}

void DeviceGroup::graphicsBufferUnmap(int bufferID)
{
  for (auto device : devices)
    device->graphicsBufferUnmap(bufferID);
}
      

void DeviceGroup::bufferResize(int bufferID, size_t newItemCount)
{
  for (auto device : devices)
    device->bufferResize(bufferID,newItemCount);
}
    
void DeviceGroup::bufferUpload(int bufferID, const void *hostPtr)
{
  for (auto device : devices)
    device->bufferUpload(bufferID,hostPtr);
}
      
void DeviceGroup::bufferUploadToSpecificDevice(int bufferID,
                                               int devID,
                                               const void *hostPtr)
{
  devices[devID]->bufferUpload(bufferID,hostPtr);
}
      


#endif

