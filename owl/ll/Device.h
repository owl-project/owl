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

#include "abstract/common.h"
#include "ll/optix.h"
#include "ll/DeviceMemory.h"

namespace owl {
  namespace ll {
    
    struct Context {
      typedef std::shared_ptr<Context> SP;
      
      Context(int owlDeviceID, int cudaDeviceID);
      ~Context();
      
      /*! linear ID (0,1,2,...) of how *we* number devices (ie,
        'first' device is alwasys device 0, no matter if it runs on
        another physical/cuda device */
      const int          owlDeviceID;
      
      /* the cuda device ID that this logical device runs on */
      const int          cudaDeviceID;

      void setActive() { CUDA_CHECK(cudaSetDevice(cudaDeviceID)); }
      
      OptixDeviceContext optixContext = nullptr;
      CUcontext          cudaContext  = nullptr;
      CUstream           stream       = nullptr;
    };
    
    struct Module {
      OptixModule module = nullptr;
    };
    struct Modules {
      std::vector<Module> modules;
    };
    
    struct ProgramGroup {
      OptixProgramGroupOptions  pgOptions = {};
      OptixProgramGroupDesc     pgDesc;
      OptixProgramGroup         pg        = nullptr;
    };
    struct ProgramGroups {
      std::vector<ProgramGroup> hitGroupPGs;
      std::vector<ProgramGroup> rayGenPGs;
      std::vector<ProgramGroup> missPGs;
    };

    struct Pipeline {
      typedef std::shared_ptr<Pipeline> SP;
      
      Pipeline(Context::SP context)
        : context(context)
      {
        pipelineLinkOptions.overrideUsesMotionBlur = false;
        pipelineLinkOptions.maxTraceDepth          = 2;
      }
      Pipeline()
      { destroy(); }
      
      void create(ProgramGroups &pgs);
      void destroy();

      Context::SP                 context;
      OptixPipeline               pipeline               = nullptr;
      OptixPipelineCompileOptions pipelineCompileOptions = {};
      OptixPipelineLinkOptions    pipelineLinkOptions    = {};
      OptixModuleCompileOptions   moduleCompileOptions   = {};
    };

    struct SBT {
      OptixShaderBindingTable sbt = {};
      DeviceMemory raygenRecordsBuffer;
      DeviceMemory missRecordsBuffer;
      DeviceMemory hitGroupRecordsBuffer;
      DeviceMemory launchParamBuffer;
    };
    struct Traversable {
      OptixTraversableHandle traversable;
    };

    struct Device {
      typedef std::shared_ptr<Device> SP;

      /*! construct a new owl device on given cuda device. throws an
          exception if for any reason that cannot be done */
      Device(int owlDeviceID, int cudaDeviceID);
      ~Device();
      
      void createPipeline()  { assert(pipeline); pipeline->create(programGroups); }
      void destroyPipeline() { assert(pipeline); pipeline->destroy(); }

      Context::SP    context;
      Pipeline::SP   pipeline;
      
      Modules        modules;
      ProgramGroups  programGroups;
      SBT            sbt;
    };
    
    struct Devices {
      typedef std::shared_ptr<Devices> SP;

      Devices(const std::vector<Device::SP> &devices);

      void createPipeline()
      { for (auto device : devices) device->createPipeline(); }
      
      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs */
      static Devices::SP create(const int *deviceIDs  = nullptr,
                                size_t     numDevices = 0);
      
      const std::vector<Device::SP> devices;
    };
    
  } // ::owl::ll
} //::owl
