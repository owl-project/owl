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

#include "pyOWL/Buffer.h"
#include "pyOWL/Context.h"

namespace pyOWL {

  Buffer::SP HostPinnedBuffer::create(Context *ctx,
                                      OWLDataType elemType,
                                      size_t elemCount)
  {
    OWLBuffer handle = owlHostPinnedBufferCreate(ctx->handle,elemType,elemCount);
    // OWLBuffer handle = owlDeviceBufferCreate(ctx->handle,elemType,elemCount,nullptr);
    return std::make_shared<HostPinnedBuffer>(handle,elemType,elemCount);
  }
  
  template<typename T, int N>
  Buffer::SP DeviceBuffer::createFromCopyable(Context *ctx,
                                              OWLDataType type,
                                              const py::buffer &buffer)
  {
    py::buffer_info info = buffer.request();
    if (info.format != py::format_descriptor<T>::format())
      throw std::runtime_error("numpy array does not match indicated owl type");

    py::array_t<T> asArray = py::cast<py::array_t<T>>(buffer);
    int numElems = asArray.shape()[0] / N;
    auto buf = asArray.request();
    const T *elems = (T*)buf.ptr;
    // PRINT(elems[0]);
    // PRINT(elems[1]);

    // const T *ptr = elems;
    // for (int i=0;i<numElems;i++) {
    //   for (int j=0;j<N;j++)
    //     std::cout << " " << *ptr++;
    //   std::cout << std::endl;
    // }
    OWLBuffer handle = owlDeviceBufferCreate(ctx->handle,type,numElems,elems);
    return std::make_shared<DeviceBuffer>(handle,type,numElems);
  }
  
  Buffer::SP DeviceBuffer::create(Context *ctx,
                                  OWLDataType type,
                                  const py::buffer &buffer)
  {
    py::buffer_info info = buffer.request();
    if (info.ndim != 1)
      // for now, support only 1D arrays ...
      throw std::runtime_error
        ("pyowl.create_device_buffer currently supports only 1D arrays");
    
    switch (type) {
    case OWL_FLOAT3: return createFromCopyable<float,3>(ctx,type,buffer);
    case OWL_INT3:   return createFromCopyable<int,3>(ctx,type,buffer);
    //   if (info.format != py::format_descriptor<float>::format())
    //     throw std::runtime_error("requested buffer type if owl.FLOAT3, but input array is not a array of floats");
    //   py::array_t<float> asFloats = py::cast<py::array_t<float>>(buffer);
    //   int numFloats = asFloats.shape()[0];
    //   auto buf = asFloats.request();
    //   const float *floats = (float*)buf.ptr;
    //   PRINT(floats[0]);
    //   PRINT(floats[1]);
    // } break;
    // case OWL_INT3: {
    //   if (info.format != py::format_descriptor<int>::format())
    //     throw std::runtime_error("requested buffer type if owl.INT3, but input array is not a array of ints");
    //   py::array_t<int> asInts = py::cast<py::array_t<int>>(buffer);
    //   int numInts = asInts.shape()[0];
    //   auto buf = asInts.request();
    //   const int *ints = (int*)buf.ptr;
    //   PRINT(ints[0]);
    //   PRINT(ints[1]);
    // } break;
    default:
      throw std::runtime_error("type "+std::to_string(type)+" not yet supported in buffercreate");
    };
    // if (info.format != py::format_descriptor<float>::format() || info.ndim != 1)
    //   throw std::runtime_error("Incompatible buffer format!");
  }
  
}
