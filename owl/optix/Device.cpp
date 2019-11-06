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

#include "optix/Context.h"
#include <optix_function_table_definition.h>

namespace optix {

  /*! a single, *GLOBAL* mutex */
  std::mutex Device::g_mutex;
  
  /*! global list of all devices. will be initialized when the first
    context gets created */
  std::vector<Device::SP> Device::g_allDevices;

  Device::SP Device::getDevice(uint32_t deviceID)
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (deviceID < g_allDevices.size())
      return g_allDevices[deviceID];
    return nullptr;
  }
  
  /*! java-style pretty-printer, for debugging */
  std::string Device::toString() 
  { return "optix::Device [cuda device #"+std::to_string(cudaDeviceID)+"]"; }

  Device::Device(int cudaDeviceID,
                 const CUcontext cudaContext,
                 const CUstream  stream,
                 const OptixDeviceContext optixContext)
    : cudaDeviceID(cudaDeviceID),
      cudaContext(cudaContext),
      stream(stream),
      optixContext(optixContext)
  {
  }

  /*! (try to) create a new device on given ID. may throw an error
    of this device can not be created (eg, if an invalid ID was
    passed, of if there is an error creating an of the optix or
    cuda device contexts */
  Device::SP Device::create(int cudaDeviceID)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDeviceID);
    std::cout << "#owl.device: - device #" << cudaDeviceID
              << ": " << prop.name << std::endl;

    CUcontext cudaContext;
    CUresult res = cuCtxCreate(&cudaContext,0,(CUdevice)cudaDeviceID);
    if (res != CUDA_SUCCESS)
      throw Error("Device::create","error in creating cuda context");
    
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    OptixDeviceContext optixContext;
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));

    Device::SP device
      = std::make_shared<Device>(cudaDeviceID,cudaContext,stream,optixContext);
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,Context::log_cb,device.get(),4));

    return device;
  }

  /*! first step in the boot-strappin process: globally initialize
    all devices */
  void Device::g_init()
  {
    std::lock_guard<std::mutex> lock(g_mutex);
    if (!g_allDevices.empty())
      // already initialized...
      return;

    // initialize cuda:
    cudaFree(0);
    
    // initialize optix:
    OPTIX_CHECK(optixInit());

    // query number of possible devices:
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw Error("Device::g_init","could not find a single CUDA capable device!?");
    
    // and create those devices:
    std::cout << "#owl.device: found " << numDevices << " CUDA-capable devices" << std::endl;
    g_allDevices.resize(numDevices);
    int numValidDevices = 0;
    for (int i=0;i<numDevices;i++) {
      try {
        g_allDevices[i] = Device::create(i);
        numValidDevices++;
      } catch (optix::Error e) {
        std::cout << "#owl.device: error in creating device #"
                  << i << ": " << e.what() << " (-> ignoring this device)" << std::endl;
      }
    }
    if (numValidDevices == 0)
      throw optix::Error("Device::g_init()","could not find *any* optix-capable device");

    std::cout << "#owl.device: found a total of " << numValidDevices << " optix-capable devices" << std::endl;
  }
    
} // ::optix

