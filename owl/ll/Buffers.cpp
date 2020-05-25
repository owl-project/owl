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

#include "owl/ll/Device.h"
#include "owl/ll/DeviceGroup.h"

namespace owl {
  namespace ll {


    // ##################################################################
    // Buffer
    // ##################################################################

    Buffer::Buffer(const size_t elementCount,
                   const size_t elementSize)
      : elementCount(elementCount),
        elementSize(elementSize)
    {
      assert(elementSize > 0);
    }

    Buffer::~Buffer()
    {
    }


    // ##################################################################
    // DeviceBuffer
    // ##################################################################

    DeviceBuffer::DeviceBuffer(const size_t elementCount,
                               const size_t elementSize)
      : Buffer(elementCount, elementSize)
    {
      devMem.alloc(elementCount*elementSize);
      d_pointer = devMem.get();
    }
    
    DeviceBuffer::~DeviceBuffer()
    {
      devMem.free();
    }

    void DeviceBuffer::resize(Device *device, size_t newElementCount) 
    {
      device->context->pushActive();

      devMem.free();
      
      this->elementCount = newElementCount;
      devMem.alloc(elementCount*elementSize);
      d_pointer = devMem.get();
      
      device->context->popActive();
    }
    
    void DeviceBuffer::upload(Device *device, const void *hostPtr) 
    {
      device->context->pushActive();
      devMem.upload(hostPtr,"DeviceBuffer::upload");
      device->context->popActive();
    }

    // ##################################################################
    // HostPinnedBuffer
    // ##################################################################

    HostPinnedBuffer::HostPinnedBuffer(const size_t elementCount,
                                       const size_t elementSize,
                                       HostPinnedMemory::SP pinnedMem)
      : Buffer(elementCount, elementSize),
        pinnedMem(pinnedMem)
    {
      d_pointer = pinnedMem->pointer;
    }


    void HostPinnedBuffer::resize(Device *device, size_t newElementCount) 
    {
      if (device->context->owlDeviceID == 0) {
      
        device->context->pushActive();
        
        pinnedMem->free();
        
        this->elementCount = newElementCount;
        pinnedMem->alloc(elementCount*elementSize);
        
        device->context->popActive();
      }
      
      d_pointer = pinnedMem->get();
    }
    
    void HostPinnedBuffer::upload(Device *device, const void *hostPtr) 
    {
      OWL_NOTIMPLEMENTED;
    }
    

    // ##################################################################
    // ManagedMemoryBuffer
    // ##################################################################

    ManagedMemoryBuffer::ManagedMemoryBuffer(const size_t elementCount,
                                       const size_t elementSize,
                                       ManagedMemory::SP managedMem)
      : Buffer(elementCount, elementSize),
        managedMem(managedMem)
    {
      d_pointer = managedMem->pointer;
    }


    void ManagedMemoryBuffer::resize(Device *device, size_t newElementCount) 
    {
      if (device->context->owlDeviceID == 0) {
      
        device->context->pushActive();
        
        managedMem->free();
        
        this->elementCount = newElementCount;
        managedMem->alloc(elementCount*elementSize);
        
        device->context->popActive();
      }
      
      d_pointer = managedMem->get();
    }
    
    void ManagedMemoryBuffer::upload(Device *device, const void *hostPtr) 
    {
      OWL_NOTIMPLEMENTED;
    }
    

    // ##################################################################
    // GraphicsBuffer
    // ##################################################################

    GraphicsBuffer::GraphicsBuffer(const size_t elementCount,
                                   const size_t elementSize,
                                   const cudaGraphicsResource_t resource)
      : Buffer(elementCount, elementSize), resource(resource)
    {

    }


    void GraphicsBuffer::resize(Device *device, size_t newElementCount)
    {
      OWL_NOTIMPLEMENTED;
    }


    void GraphicsBuffer::upload(Device *device, const void  *hostPtr)
    {
      OWL_NOTIMPLEMENTED;
    }


    void GraphicsBuffer::map(Device *device, CUstream stream)
    {
      CUDA_CHECK(cudaGraphicsMapResources(1, &resource, stream));
      size_t size = 0;
      CUDA_CHECK(cudaGraphicsResourceGetMappedPointer(&d_pointer, &size, resource));
      if (elementCount * elementSize != size)
      {
        throw std::runtime_error("mapped resource has unexpected size");
      }
    }


    void GraphicsBuffer::unmap(Device *device, CUstream stream)
    {
      CUDA_CHECK(cudaGraphicsUnmapResources(1, &resource, stream));
      d_pointer = nullptr;
    }
    

  } // ::owl::ll
} // ::owl

