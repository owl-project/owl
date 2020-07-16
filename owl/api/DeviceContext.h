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

#include "owl/owl.h"
#include "owl/ll/common.h"
#include "owl/ll/DeviceMemory.h"
#include "owl/ll/helper/optix.h"
// ll
// #include "../ll/DeviceGroup.h"

namespace owl {

  struct RangeAllocator {
    int alloc(size_t size);
    void release(size_t begin, size_t size);
    size_t maxAllocedID = 0;
  private:
    struct FreedRange {
      size_t begin;
      size_t size;
    };
    std::vector<FreedRange> freedRanges;
  };

  struct SBT {
    size_t rayGenRecordCount   = 0;
    size_t rayGenRecordSize    = 0;
    DeviceMemory rayGenRecordsBuffer;

    size_t hitGroupRecordSize  = 0;
    size_t hitGroupRecordCount = 0;
    DeviceMemory hitGroupRecordsBuffer;

    size_t missProgRecordSize  = 0;
    size_t missProgRecordCount = 0;
    DeviceMemory missProgRecordsBuffer;

    DeviceMemory launchParamsBuffer;
  };

  /*! what will eventually containt the whole owl context across all gpus */
  struct Context;

  /*! optix and cuda context for a single, specific GPU */
  struct DeviceContext {
    typedef std::shared_ptr<DeviceContext> SP;
    
    DeviceContext(Context *parent,
                  int owlID,
                  int cudaID);
    
    void buildPrograms();
    void buildMissPrograms();
    void buildRayGenPrograms();
    void buildHitGroupPrograms();
    // void buildIsecPrograms();
    // // void buildBoundsPrograms();
    // void buildAnyHitPrograms();
    // void buildClosestHitPrograms();
    void destroyPrograms();
    void destroyPipeline();
    void buildPipeline();
      
    std::vector<OptixProgramGroup> allActivePrograms;

    /*! helper function - return cuda name of this device */
    std::string getDeviceName() const;
      
    /*! helper function - return cuda device ID of this device */
    int getCudaDeviceID() const;

    CUstream getStream() const { return stream; }
    
    OptixDeviceContext optixContext = nullptr;
    CUcontext          cudaContext  = nullptr;
    CUstream           stream       = nullptr;

    /*! sets the pipelineCompileOptions etc based on
      maxConfiguredInstanceDepth */
    void configurePipelineOptions();
      
    OptixPipelineCompileOptions pipelineCompileOptions = {};
    OptixPipelineLinkOptions    pipelineLinkOptions    = {};
    OptixModuleCompileOptions   moduleCompileOptions   = {};
    OptixPipeline               pipeline               = nullptr;
    SBT                         sbt;

    /* the cuda device ID that this logical device runs on */
    const int          cudaDeviceID;

    /*! linear ID (0,1,2,...) of how *we* number devices (i.e.,
      'first' device is always device 0, no matter if it runs on
      another physical/cuda device) */
    // const int          owlDeviceID;
    const int ID;
      
    Context    *const parent;
  };

  std::vector<DeviceContext::SP> createDeviceContexts(int32_t *requestedDeviceIDs,
                                                      int      numRequestedDevices);

  /*! helper class that will set the active cuda device (to the device
      associated with a given Context::DeviceData) for the duration fo
      the lifetime of this class, and resets it to whatever it was
      after class dies */
  struct SetActiveGPU {
    SetActiveGPU(DeviceContext::SP device)
    {
      CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
      CUDA_CHECK(cudaSetDevice(device->cudaDeviceID));
    }
    ~SetActiveGPU()
    {
      CUDA_CHECK_NOTHROW(cudaSetDevice(savedActiveDeviceID));
    }
  private:
    int savedActiveDeviceID = -1;
  };
  
} // ::owl

