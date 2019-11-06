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

#pragma once

#include "optix/Base.h"

namespace optix {

  /*! captures all device-specific contexts - cuda context, device
    context, etc - that other functions will then run on. none of the
    members of this class should be modified by other classes (other
    than the mutex, of course) */
  struct Device : public CommonBase {
    typedef std::shared_ptr<Device> SP;

    /*! (try to) create a new device on given ID. may throw an error
        of this device can not be created (eg, if an invalid ID was
        passed, of if there is an error creating an of the optix or
        cuda device contexts */
    static Device::SP create(int deviceID);

    Device(int cudaDeviceID,
           const CUcontext cudaContext,
           const CUstream  cudaStream,
           const OptixDeviceContext optixContext);

    /*! create a list of all eligible devices, each fully initialized
        (if possible), or represented with a nullptr, if not */
    static std::vector<Device::SP> queryAllDevices();

    /*! java-style pretty-printer, for debugging */
    virtual std::string toString() override;
      
    /*! first step in the boot-strappin process: globally initialize
        all devices */
    static void g_init();
    
    /*! perform a cudaSetActive on this device (for given thread) */
    void setActive();

    /*! allows for mutex'ing different operations that run
        concurrently on this device */
    std::mutex mutex;
    
    const int                         cudaDeviceID;

    /*! a cuda context for the given device */
    const CUcontext                   cudaContext;
    
    /*! a stream we create for this cude context; you can of course
        use other streams as well, but this should be the default
        stream to be used for this device (in order to allow for this
        device to work independently of other devices */
    const CUstream                    stream;

    /*! the (low-level!) optix device context for this device - NOT to
        be confused with the (high-level) optix::Context created by
        this library */
    const OptixDeviceContext          optixContext;


    /*! a single, *GLOBAL* mutex */
    static std::mutex g_mutex;
    
    /*! global list of all devices. will be initialized when the first
        context gets created */
    static std::vector<Device::SP> g_allDevices;
  };

} // ::optix
