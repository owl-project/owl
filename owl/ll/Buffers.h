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

#pragma once

#include "owl/ll/helper/optix.h"
#include "owl/ll/DeviceMemory.h"
#include "owl/ll/DeviceGroup.h"

namespace owl {
  namespace ll {

    /*! base abstraction for any buffer type - buffers are device
        memory that is "known to" (and usually, but now always,
        managed by) owl; unlike optix 6 we explicitly expose the
        differnet types of device memory supported by CDUA; eg, a
        "DeviceBuffer" corresponds to regular cudaMalloc memory, a
        HostPinnedBuffer corresponds to CudaMallocHost,
        ManagedMemBuffer corresponds to cudaMallocManaged, etc. */
    struct Buffer 
    {
      Buffer(const size_t elementCount,
             const size_t elementSize);
      virtual ~Buffer();
      inline void *get() const { return d_pointer; }
      virtual void resize(Device *device, size_t newElementCount) = 0;
      virtual void upload(Device *device, const void *hostPtr) = 0;
      
      size_t       elementCount;
      size_t       elementSize;
      void        *d_pointer = nullptr;
    };

    /*! a buffer of regular device memory alloced with cudaMalloc. In
        a multi-GPU setting this buffer will be created once per
        device; ie, each device will have its owl local buffer, and
        all objects on this device that use this buffer will see
        _this_ device's buffer address. */
    struct DeviceBuffer : public Buffer
    {
      DeviceBuffer(const size_t elementCount,
                   const size_t elementSize);
      ~DeviceBuffer();
      void resize(Device *device, size_t newElementCount) override;
      void upload(Device *device, const void *hostPtr) override;
      DeviceMemory devMem;
    };
    
    /*! buffer type that corresponds to CUDA's "host pinned memory"
        (see cudaMallocHost). Host pinned memory gets pinned on the
        host, but is accessible on all devices; device-accesses to
        such memory will likely be significantly slower than those to
        device buffers, but unlike device buffers different devices
        can "see" (and read from/write to) the _same_ memory across
        all devices. Unlike device buffers pinned buffers span across
        all devices, and are therefore allocated/refcounted in
        DeviceGroup, with this class just being an adapter to this
        cross-devicegroup owned object */
    struct HostPinnedBuffer : public Buffer
    {
      HostPinnedBuffer(const size_t elementCount,
                       const size_t elementSize,
                       HostPinnedMemory::SP pinnedMem);
      void resize(Device *device, size_t newElementCount) override;
      void upload(Device *device, const void *hostPtr) override;

      /*! refcounted pointed to the class (created and owned by the
          DeviceGroup) that managed the actual storage of this
          meomry. */
      HostPinnedMemory::SP pinnedMem;
    };
      
    /*! buffer type that corresponds to CUDA's "managed memory" (see
      cudaMallocManaged). See CUDA's doc on managed memory. */
    struct ManagedMemoryBuffer : public Buffer
    {
      ManagedMemoryBuffer(const size_t elementCount,
                          const size_t elementSize,
                          ManagedMemory::SP pinnedMem);
      void resize(Device *device, size_t newElementCount) override;
      void upload(Device *device, const void *hostPtr) override;
      
      /*! refcounted pointed to the class (created and owned by the
          DeviceGroup) that managed the actual storage of this
          meomry. */
      ManagedMemory::SP managedMem;
    };

    /*! buffer type wrapping CUDA graphics resources */
    struct GraphicsBuffer : public Buffer
    {
      GraphicsBuffer(const size_t elementCount,
                     const size_t elementSize,
                     const cudaGraphicsResource_t resource);

      void resize(Device *device, size_t newElementCount) override;
      void upload(Device *device, const void *hostPtr) override;

      void map(Device *device, CUstream stream);
      void unmap(Device *device, CUstream stream);

      cudaGraphicsResource_t resource;
    };
      
  } // ::owl::ll
} // ::owl
