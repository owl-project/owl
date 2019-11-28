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
      : context(std::make_shared<Context>(owlDeviceID,cudaDeviceID)),
        pipeline(std::make_shared<Pipeline>(context))
    {
      std::cout << "#owl.ll: successfully created owl device #" << owlDeviceID
                << " on CUDA device #" << cudaDeviceID << std::endl;
    }
    

    Device::~Device()
    {
      destroyPipeline();
    }

    void Pipeline::destroy()
    {
      if (pipeline) {
        context->setActive();
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
      }
    }
    
    void Pipeline::create(ProgramGroups &pgs)
    {
      destroy();
      
      context->setActive();
      std::vector<OptixProgramGroup> allPGs;
      for (auto &pg : pgs.rayGenPGs)
        allPGs.push_back(pg.pg);
      for (auto &pg : pgs.hitGroupPGs)
        allPGs.push_back(pg.pg);
      for (auto &pg : pgs.missPGs)
        allPGs.push_back(pg.pg);
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      
      OPTIX_CHECK(optixPipelineCreate(context->optixContext,
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

