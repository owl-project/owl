// // ======================================================================== //
// // Copyright 2018 Ingo Wald                                                 //
// //                                                                          //
// // Licensed under the Apache License, Version 2.0 (the "License");          //
// // you may not use this file except in compliance with the License.         //
// // You may obtain a copy of the License at                                  //
// //                                                                          //
// //     http://www.apache.org/licenses/LICENSE-2.0                           //
// //                                                                          //
// // Unless required by applicable law or agreed to in writing, software      //
// // distributed under the License is distributed on an "AS IS" BASIS,        //
// // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// // See the License for the specific language governing permissions and      //
// // limitations under the License.                                           //
// // ======================================================================== //

// #pragma once

// // cuda
// // #include <cuda.h>
// #include <cuda_runtime.h>
// // ours
// #include "gdt.h"
// // std
// #include <vector>
// #include <sstream>

// namespace gdt {
  
// #define CUDA_CALL(call)							\
//     {									\
//       cudaError_t rc = cuda##call;                                      \
//       if (rc != cudaSuccess) {                                          \
//         std::stringstream txt;                                          \
//         cudaError_t err =  rc; /*cudaGetLastError();*/                  \
//         txt << "CUDA Error " << cudaGetErrorName(err)                   \
//             << " (" << cudaGetErrorString(err) << ")";                  \
//         throw std::runtime_error(txt.str());                            \
//       }                                                                 \
//     }

// #define CUDA_CALL_NOEXCEPT(call)                                        \
//     {									\
//       cuda##call;                                                       \
//     }

//     /*! abstraction of a CUDA platform" of one or more cuda-enabled
//         devices */
//     struct Platform { 
//       Platform()
//       {
//         // std::cout << "initializing CUDA" << std::endl;
//         // CUDA_CALL(cuInit(0));
//         // cuDeviceGetCount(&numDevices);
//         cudaGetDeviceCount(&numDevices);
//         PRINT(numDevices);
//       }
    
//       ~Platform()
//       {
//       }
    
//       int numDevices;
//     };


//     // /*! \brief abstraction for a device-side buffer of a given type and size

//     //   device-side buffer means that this buffer exists only on the
//     //   device, and cannot be accessed directly on the host
//     //  */
//     // template<typename T>
//     // struct DevBuffer {
//     //   DevBuffer(size_t numElements)
//     //   {
//     //     actual = std::make_shared<Actual>(numElements);
//     //   }
      
//     //   size_t size() const { return actual->numElements; }
//     //   T     *ptr()  const { return actual->data; }

//     // private:
//     //   /*! the actual buffer, which gets allocated only once, and is
//     //       then shared via a shared_ptr */
//     //   struct Actual {
//     //     Actual(size_t numElements)
//     //       : numElements(numElements),
//     //         data(nullptr)
//     //     {
//     //       CUDA_CALL(Malloc((void **)&data,numElements*sizeof(T)));
//     //       PRINT(data);
//     //     }
//     //     ~Actual()
//     //     {
//     //       if (data) CUDA_CALL_NOEXCEPT(Free(data));
//     //       data = nullptr;
//     //     }
//     //     T  *data;
//     //     size_t numElements;
//     //   };
//     //   std::shared_ptr<Actual> actual;
//     // };

//     // /*! \brief abstraction for a host-side buffer of a given type and size

//     //   device-side buffer means that this buffer exists only on the
//     //   host, and cannot be accessed directly on the device
//     //  */
//     // template<typename T>
//     // struct HostBuffer {
//     //   HostBuffer(size_t numElements)
//     //     : actual(std::make_shared<std::vector<T>>(numElements))
//     //   {}
//     //   size_t size() const { return actual->size(); }
//     //   T *ptr() const { return &(*actual)[0]; }

//     //   T &operator[](size_t idx) { PRINT(&(*actual)[idx]); return (*actual)[idx]; }
//     // private:
//     //   std::shared_ptr<std::vector<T>> actual;
//     // };

//     // template<typename T>
//     // void copy(HostBuffer<T> &host, DevBuffer<T> &dev);

//     // template<typename T>
//     // HostBuffer<T> download(DevBuffer<T> &dev)
//     // {
//     //   HostBuffer<T> host(dev.size());
//     //   PRINT(host.ptr());
//     //   PRINT(dev.ptr());
//     //   PRINT(dev.size()*sizeof(T));
//     //   CUDA_CALL(Memcpy(host.ptr(),dev.ptr(),dev.size()*sizeof(T),cudaMemcpyDeviceToHost));
//     //   return host;
//     // }
    
// }


    
