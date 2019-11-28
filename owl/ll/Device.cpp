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

#include "Device.h"
#include <optix_function_table_definition.h>

namespace owl {
  namespace ll {

    static void context_log_cb(unsigned int level,
                               const char *tag,
                               const char *message,
                               void *)
    {
      fprintf( stderr, "[%2d][%12s]: %s\n", level, tag, message );
    }
  
    /*! construct a new owl device on given cuda device. throws an
      exception if for any reason that cannot be done */
    Context::Context(int owlDeviceID,
                     int cudaDeviceID)
      : owlDeviceID(owlDeviceID),
        cudaDeviceID(cudaDeviceID)
    {
      std::cout << "#owl.ll: trying to create owl device on CUDA device #"
                << cudaDeviceID << std::endl;

      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, cudaDeviceID);
      std::cout << "#owl.ll: - device: " << prop.name << std::endl;

      CUDA_CHECK(cudaSetDevice(cudaDeviceID));
      CUDA_CHECK(cudaStreamCreate(&stream));
      
      CUresult  cuRes = cuCtxGetCurrent(&cudaContext);
      if (cuRes != CUDA_SUCCESS) 
        throw std::runtime_error("Error querying current CUDA context...");
      
      OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
      OPTIX_CHECK(optixDeviceContextSetLogCallback
                  (optixContext,context_log_cb,this,4));
    }

    /*! construct a new owl device on given cuda device. throws an
      exception if for any reason that cannot be done */
    Context::~Context()
    {
      std::cout << "#owl.ll: destroying owl device #"
                << owlDeviceID
                << " on CUDA device #" 
                << cudaDeviceID << std::endl;
    }

    
    
    /*! construct a new owl device on given cuda device. throws an
      exception if for any reason that cannot be done */
    Device::Device(int owlDeviceID, int cudaDeviceID)
      : context(std::make_shared<Context>(owlDeviceID,cudaDeviceID))
    {
      std::cout << "#owl.ll: successfully created owl device #" << owlDeviceID
                << " on CUDA device #" << cudaDeviceID << std::endl;
    }
    

    Device::~Device()
    {
      destroyPipeline();
    }

    void Context::destroyPipeline()
    {
      if (pipeline) {
        setActive();
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
      }
    }

    void Device::setHitGroupClosestHit(int pgID,
                                       int moduleID,
                                       const char *progName)
    {
      assert(pgID >= 0);
      assert(pgID < hitGroupPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      hitGroupPGs[pgID].closestHit.moduleID = moduleID;
      hitGroupPGs[pgID].closestHit.progName = progName;
    }
    
    void Device::setRayGenPG(int pgID, int moduleID, const char *progName)
    {
      assert(pgID >= 0);
      assert(pgID < rayGenPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      rayGenPGs[pgID].program.moduleID = moduleID;
      rayGenPGs[pgID].program.progName = progName;
    }
    
    void Device::setMissPG(int pgID, int moduleID, const char *progName)
    {
      assert(pgID >= 0);
      assert(pgID < missPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      missPGs[pgID].program.moduleID = moduleID;
      missPGs[pgID].program.progName = progName;
    }
    
    /*! will destroy the *optix handles*, but will *not* clear the
        modules vector itself */
    void Modules::destroyOptixHandles(Context *context)
    {
      for (auto &module : modules) {
        if (module.module != nullptr) {
          optixModuleDestroy(module.module);
          module.module = nullptr;
        }
      }
    }


    void Modules::buildOptixHandles(Context *context)
    {
      assert(!modules.empty());
      std::cout << "#owl.ll(" << context->owlDeviceID << "): "
                << "building " << modules.size() << " modules" << std::endl;
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      for (int moduleID=0;moduleID<modules.size();moduleID++) {
        Module &module = modules[moduleID];
        assert(module.module == nullptr);
        OPTIX_CHECK(optixModuleCreateFromPTX(context->optixContext,
                                             &context->moduleCompileOptions,
                                             &context->pipelineCompileOptions,
                                             module.ptxCode,
                                             strlen(module.ptxCode),
                                             nullptr,      // Log string
                                             0,            // Log string sizse
                                             &module.module
                                             ));
        assert(module.module != nullptr);
        std::cout << "#owl.ll: created module #" << moduleID << std::endl;
      }
    }

    void Modules::alloc(size_t count)
    {
      assert(modules.empty());
      modules.resize(count);
    }

    void Modules::set(size_t slot, const char *ptxCode)
    {
      assert(!modules.empty());
      
      assert(slot >= 0);
      assert(slot < modules.size());

      assert(!modules[slot].module);
      modules[slot].ptxCode = ptxCode;
    }

    void Device::buildOptixPrograms()
    {
      OptixProgramGroupOptions pgOptions = {};

      for (int pgID=0;pgID<rayGenPGs.size();pgID++) {
        RayGenPG &pg     = rayGenPGs[pgID];
        Module   *module = modules.get(pg.program.moduleID);
        OptixProgramGroupDesc pgDesc    = {};
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        if (module) {
          assert(module->module != nullptr);
          assert(pg.program.progName != nullptr);
          pgDesc.raygen.module            = module->module;
          pgDesc.raygen.entryFunctionName = pg.program.progName;
        } else {
          pgDesc.raygen.module            = nullptr;
          pgDesc.raygen.entryFunctionName = nullptr;
        }
        char log[2048];
        size_t sizeof_log = sizeof( log );
        OPTIX_CHECK(optixProgramGroupCreate(context->optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &pg.pg
                                            ));
      }
      
    }
    
    void Device::destroyOptixPrograms()
    {
      PING;
    }

    void Device::allocHitGroupPGs(size_t count)
    {
      assert(hitGroupPGs.empty());
      hitGroupPGs.resize(count);
    }
    
    void Device::allocRayGenPGs(size_t count)
    {
      assert(rayGenPGs.empty());
      rayGenPGs.resize(count);
    }
    
    void Device::allocMissPGs(size_t count)
    {
      assert(missPGs.empty());
      missPGs.resize(count);
    }
      

    void Context::createPipeline(Device *device)
    {
      if (pipeline != nullptr)
        throw std::runtime_error("pipeline already created!?");
      
      setActive();
      std::vector<OptixProgramGroup> allPGs;
      assert(!device->rayGenPGs.empty());
      for (auto &pg : device->rayGenPGs)
        allPGs.push_back(pg.pg);
      assert(!device->hitGroupPGs.empty());
      for (auto &pg : device->hitGroupPGs)
        allPGs.push_back(pg.pg);
      assert(!device->missPGs.empty());
      for (auto &pg : device->missPGs)
        allPGs.push_back(pg.pg);

      if (allPGs.empty())
        throw std::runtime_error("trying to create a pipeline w/ 0 programs!?");
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      
      OPTIX_CHECK(optixPipelineCreate(optixContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      allPGs.data(),
                                      allPGs.size(),
                                      log,&sizeof_log,
                                      &pipeline
                                      ));
      OPTIX_CHECK(optixPipelineSetStackSize
                  (pipeline,
                   /* [in] The pipeline to configure the stack size for */
                   2*1024,
                   /* [in] The direct stack size requirement for
                      direct callables invoked from IS or AH. */
                   2*1024,
                   /* [in] The direct stack size requirement for
                      direct callables invoked from RG, MS, or CH.  */
                   2*1024,
                   /* [in] The continuation stack requirement. */
                   3
                   /* [in] The maximum depth of a traversable graph
                      passed to trace. */
                   ));
    }
      


    
    /* create an instance of this object that has properly
       initialized devices */
    Devices::SP Devices::create(const int *deviceIDs,
                                size_t     numDevices)
    {
      assert((deviceIDs == nullptr && numDevices == 0)
             ||
             (deviceIDs != nullptr && numDevices > 0));

      // ------------------------------------------------------------------
      // init cuda, and error-out if no cuda devices exist
      // ------------------------------------------------------------------
      std::cout << "#owl.ll: initializing CUDA" << std::endl;
      cudaFree(0);
      
      int totalNumDevices = 0;
      cudaGetDeviceCount(&totalNumDevices);
      if (totalNumDevices == 0)
        throw std::runtime_error("#owl.ll: no CUDA capable devices found!");
      std::cout << "#owl.ll: found " << totalNumDevices << " CUDA device(s)" << std::endl;

      
      // ------------------------------------------------------------------
      // init optix itself
      // ------------------------------------------------------------------
      std::cout << "#owl.ll: initializing optix 7" << std::endl;
      OPTIX_CHECK(optixInit());

      // ------------------------------------------------------------------
      // check if a device ID list was passed, and if not, create one
      // ------------------------------------------------------------------
      std::vector<int> allDeviceIDs;
      if (deviceIDs == nullptr) {
        for (int i=0;i<totalNumDevices;i++) allDeviceIDs.push_back(i);
        numDevices = allDeviceIDs.size();
        deviceIDs  = allDeviceIDs.data();
      }
      // from here on, we need a non-empty list of requested device IDs
      assert(deviceIDs != nullptr && numDevices > 0);
      
      // ------------------------------------------------------------------
      // create actual devices, ignoring those that failed to initialize
      // ------------------------------------------------------------------
      std::vector<Device::SP> devices;
      for (int i=0;i<numDevices;i++) {
        try {
          Device::SP dev = std::make_shared<Device>(devices.size(),deviceIDs[i]);
          assert(dev);
          devices.push_back(dev);
        } catch (std::exception &e) {
          std::cout << "#owl.ll: Error creating optix device on CUDA device #"
                    << deviceIDs[i] << ": " << e.what() << " ... dropping this device"
                    << std::endl;
        }
      }

      // ------------------------------------------------------------------
      // some final sanity check that we managed to create at least
      // one device...
      // ------------------------------------------------------------------
      if (devices.empty())
        throw std::runtime_error("fatal error - could not find/create any optix devices");
      
      return std::make_shared<Devices>(devices);
    }

    Devices::Devices(const std::vector<Device::SP> &devices)
      : devices(devices)
    {
      assert(!devices.empty());
      std::cout << "#owl.ll: created device group with "
                << devices.size() << " device(s)" << std::endl;
    }
    
  } // ::owl::ll
} //::owl

