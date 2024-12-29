// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "owl/helper/cuda.h"

namespace owl {

  struct DeviceMemory {
    inline ~DeviceMemory() { if (!externallyManaged && !empty()) free(); }
    inline bool   alloced()  const { return !empty(); }
    inline bool   empty()    const { return sizeInBytes == 0; }
    inline bool   notEmpty() const { return !empty(); }
    inline size_t size()     const { return sizeInBytes; }
    
    inline void alloc(size_t size);
    inline void allocManaged(size_t size);
    inline void *get();
    inline void upload(const void *h_pointer, const char *debugMessage = nullptr);
    inline void uploadAsync(const void *h_pointer, cudaStream_t stream);
    inline void uploadAsync(const void *h_pointer, size_t offset, size_t size, cudaStream_t stream);
    inline void download(void *h_pointer);
    inline void free();
    template<typename T>
    inline void upload(const std::vector<T> &vec);
      
    size_t      sizeInBytes { 0 };
    CUdeviceptr d_pointer   { 0 };
    bool externallyManaged = false;
  };

  inline void DeviceMemory::alloc(size_t size)
  {
    if (size == sizeInBytes) return;
    
    assert(!externallyManaged);
    if (alloced()) free();
      
    assert(empty());
    this->sizeInBytes = size;
    if (size == 0)
      d_pointer = 0;
    else
      OWL_CUDA_CHECK(cudaMalloc( (void**)&d_pointer, sizeInBytes));
    assert(alloced() || size == 0);
  }
    
  inline void DeviceMemory::allocManaged(size_t size)
  {
    assert(!externallyManaged);
    assert(empty());
    if (alloced()) free();
    this->sizeInBytes = size;
    if (size == 0)
      d_pointer = 0;
    else
      OWL_CUDA_CHECK(cudaMallocManaged( (void**)&d_pointer, sizeInBytes));
    assert(alloced() || size == 0);
  }
    
  inline void *DeviceMemory::get()
  {
    return (void*)d_pointer;
  }

  inline void DeviceMemory::upload(const void *h_pointer, const char *debugMessage)
  {
    assert(alloced() || empty());
    OWL_CUDA_CHECK2(debugMessage,
                cudaMemcpy((void*)d_pointer, h_pointer,
                           sizeInBytes, cudaMemcpyHostToDevice));
  }
    
  inline void DeviceMemory::uploadAsync(const void *h_pointer, cudaStream_t stream)
  {
    assert(alloced() || empty());
    OWL_CUDA_CHECK(cudaMemcpyAsync((void*)d_pointer, h_pointer,
                               sizeInBytes, cudaMemcpyHostToDevice,
                               stream));
  }

  inline void DeviceMemory::uploadAsync(const void *h_pointer, size_t offset, size_t size, cudaStream_t stream)
  {
    assert(alloced() || empty());
    assert(offset + size <= sizeInBytes);
    OWL_CUDA_CHECK(cudaMemcpyAsync((void*)(d_pointer + offset), h_pointer,
                               size, cudaMemcpyHostToDevice,
                               stream));
  }
    
  inline void DeviceMemory::download(void *h_pointer)
  {
    assert(alloced() || sizeInBytes == 0);
    OWL_CUDA_CHECK(cudaMemcpy(h_pointer, (void*)d_pointer, 
                          sizeInBytes, cudaMemcpyDeviceToHost));
  }
    
  inline void DeviceMemory::free()
  {
    assert(!externallyManaged);
    assert(alloced() || empty());
    if (!empty()) {
      OWL_CUDA_CHECK(cudaFree((void*)d_pointer));
    }
    sizeInBytes = 0;
    d_pointer   = 0;
    assert(empty());
  }

  template<typename T>
  inline void DeviceMemory::upload(const std::vector<T> &vec)
  {
    if (!alloced()) {
      alloc(vec.size()*sizeof(T));
    } else {
      assert(size() == vec.size()*sizeof(T));
    }
    assert(alloced() || vec.empty());
    upload(vec.data());
  }
    
  struct PinnedHostMem {
    PinnedHostMem() = default;
    ~PinnedHostMem() { if (ptr) cudaFree(ptr); }
    void resize(size_t N) {
      if (ptr) cudaFree(ptr);
      ptr = 0;
      if (N > 0) cudaMallocHost(&ptr,N);
    }
    uint8_t *data() { return ptr; }
    
    uint8_t *ptr = 0;
  };
  
} //::owl
