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
    struct ProgramGroups;
    struct Device;
    
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

      void createPipeline(Device *device);
      void destroyPipeline();
      
      OptixDeviceContext optixContext = nullptr;
      CUcontext          cudaContext  = nullptr;
      CUstream           stream       = nullptr;

      OptixPipelineCompileOptions pipelineCompileOptions = {};
      OptixPipelineLinkOptions    pipelineLinkOptions    = {};
      OptixModuleCompileOptions   moduleCompileOptions   = {};
      OptixPipeline               pipeline               = nullptr;
    };
    
    struct Module {
      OptixModule module = nullptr;
      const char *ptxCode;
      void create(Context *context);
    };
    struct Modules {
      ~Modules() {
        assert(noActiveHandles());
      }
      bool noActiveHandles() {
        for (auto &module : modules) if (module.module != nullptr) return false;
        return true;
      }
      size_t size() const { return modules.size(); }
      void alloc(size_t size);
      /*! will destroy the *optix handles*, but will *not* clear the
        modules vector itself */
      void destroyOptixHandles(Context *context);
      void buildOptixHandles(Context *context);
      void set(size_t slot, const char *ptxCode);
      
      std::vector<Module> modules;
    };
    
    struct ProgramGroup {
      OptixProgramGroupOptions  pgOptions = {};
      OptixProgramGroupDesc     pgDesc;
      OptixProgramGroup         pg        = nullptr;
    };
    struct Program {
      const char *progName = nullptr;
      int         moduleID = -1;
    };
    struct RayGenPG : public ProgramGroup {
      Program program;
    };
    struct MissPG : public ProgramGroup {
      Program program;
    };
    struct HitGroupPG : public ProgramGroup {
      Program anyHit;
      Program closestHit;
      Program intersect;
    };
    // struct ProgramGroups {
    //   std::vector<HitGroupPG> hitGroupPGs;
    //   std::vector<RayGenPG> rayGenPGs;
    //   std::vector<MissPG> missPGs;
    // };

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

      void createPipeline()  { context->createPipeline(this); }
      void destroyPipeline()
      {
        context->destroyPipeline();
      }

      void rebuildModules()
      {
        // modules shouldn't be rebuilt while a pipeline is still using them(?)
        assert(context->pipeline == nullptr);
        modules.destroyOptixHandles(context.get());
        modules.buildOptixHandles(context.get());
      }

      void setHitGroupClosestHit(int pgID, int moduleID, const char *progName);
      void setRayGenPG(int pgID, int moduleID, const char *progName);
      void setMissPG(int pgID, int moduleID, const char *progName);
      
      void allocModules(size_t count)
      { modules.alloc(count); }
      void allocHitGroupPGs(size_t count);
      void allocRayGenPGs(size_t count);
      void allocMissPGs(size_t count);
      
      Context::SP             context;
      
      Modules                 modules;
      std::vector<HitGroupPG> hitGroupPGs;
      std::vector<RayGenPG>   rayGenPGs;
      std::vector<MissPG>     missPGs;
      SBT                     sbt;
    };
    
    struct Devices {
      typedef std::shared_ptr<Devices> SP;

      Devices(const std::vector<Device::SP> &devices);

      void allocModules(size_t count)
      { for (auto device : devices) device->allocModules(count); }
      void setModule(size_t slot, const char *ptxCode)
      { for (auto device : devices) device->modules.set(slot,ptxCode); }
      void rebuildModules()
      {
        for (auto device : devices)
          device->rebuildModules();
      }
      void createPipeline()
      { for (auto device : devices) device->createPipeline(); }
      
      void allocHitGroupPGs(size_t count)
      { for (auto device : devices) device->allocHitGroupPGs(count); }
      void allocRayGenPGs(size_t count)
      { for (auto device : devices) device->allocRayGenPGs(count); }
      void allocMissPGs(size_t count)
      { for (auto device : devices) device->allocMissPGs(count); }

      void setHitGroupClosestHit(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setHitGroupClosestHit(pgID,moduleID,progName); }
      void setRayGenPG(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setRayGenPG(pgID,moduleID,progName); }
      void setMissPG(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setMissPG(pgID,moduleID,progName); }
      
      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs */
      static Devices::SP create(const int *deviceIDs  = nullptr,
                                size_t     numDevices = 0);
      
      const std::vector<Device::SP> devices;
    };
    
  } // ::owl::ll
} //::owl
