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

#include "LaunchParams.h"
#include "../ll/Device.h"
#include "Context.h"

namespace owl {

  LaunchParamsType::LaunchParamsType(Context *const context,
                                     size_t varStructSize,
                                     const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->launchParamTypes,varStructSize,varDecls)
  {
  }
  
  LaunchParams::LaunchParams(Context *const context,
                 LaunchParamsType::SP type) 
    : SBTObject(context,context->launchParams,type)
  {
    assert(context);
    assert(type);
    assert(type.get());
  }

  LaunchParams::DeviceData::DeviceData(size_t  dataSize,
                                       Device *device)
    : device(device),
      dataSize(dataSize)
  {
    int oldActive = device->pushActive();
    CUDA_CHECK(cudaStreamCreate(&stream));
    deviceMemory.alloc(dataSize);
    hostMemory.resize(dataSize);
    device->popActive(oldActive);
  }

  CUstream LaunchParams::getCudaStream(int deviceID)
  {
    return getDD(deviceID).stream;
  }

  void LaunchParams::sync()
  {
    for (auto device : context->llo->devices) {
      int oldActive = device->context->pushActive();
      cudaStreamSynchronize(getCudaStream(device->ID));
      device->context->popActive(oldActive);
    }
  }

    // /*! launch *with* launch params */
    // void Device::launch(int rgID,
    //                     const vec2i &dims,
    //                     int32_t launchParamsID,
    //                     LLOWriteLaunchParamsCB writeLaunchParamsCB,
    //                     const void *cbData)
    // {
    //   // STACK_PUSH_ACTIVE(context);
    //   // LaunchParams *lp
    //   //   = checkGetLaunchParams(launchParamsID);
      
    //   // // call the callback to generate the host-side copy of the
    //   // // launch params struct
    //   // writeLaunchParamsCB(lp->hostMemory.data(),context->owlDeviceID,cbData);
      
    //   // lp->deviceMemory.uploadAsync(lp->hostMemory.data(),
    //   //                              lp->stream);
    //   // assert("check valid launch dims" && dims.x > 0);
    //   // assert("check valid launch dims" && dims.y > 0);
    //   // assert("check valid ray gen program ID" && rgID >= 0);
    //   // assert("check valid ray gen program ID" && rgID <  rayGenPGs.size());

    //   // assert("check raygen records built" && sbt.rayGenRecordCount != 0);
    //   // OptixShaderBindingTable localSBT = {};
    //   localSBT.raygenRecord
    //     = (CUdeviceptr)addPointerOffset(sbt.rayGenRecordsBuffer.get(),
    //                                     rgID * sbt.rayGenRecordSize);

    //   if (!sbt.missProgRecordsBuffer.alloced() &&
    //       !sbt.hitGroupRecordsBuffer.alloced()) {
    //     // Apparently this program does not have any hit records *or*
    //     // miss records, which means either something's horribly wrong
    //     // in the app, or this is more cuda-style "raygen-only" launch
    //     // (i.e., a launch of a raygen program that doesn't actually trace
    //     // any rays). If the latter, let's "fake" a valid SBT by
    //     // writing in some (senseless) values to not trigger optix's
    //     // own sanity checks.
    //     static WarnOnce warn("launching an optix pipeline that has neither miss nor hitgroup programs set. This may be OK if you *only* have a raygen program, but is usually a sign of a bug - please double-check");
    //     localSBT.missRecordBase
    //       = (CUdeviceptr)32;
    //     localSBT.missRecordStrideInBytes
    //       = (uint32_t)32;
    //     localSBT.missRecordCount
    //       = 1;

    //     localSBT.hitgroupRecordBase
    //       = (CUdeviceptr)32;
    //     localSBT.hitgroupRecordStrideInBytes
    //       = (uint32_t)32;
    //     localSBT.hitgroupRecordCount
    //       = 1;
    //   } else {
    //     assert("check miss records built" && sbt.missProgRecordCount != 0);
    //     localSBT.missRecordBase
    //       = (CUdeviceptr)sbt.missProgRecordsBuffer.get();
    //     localSBT.missRecordStrideInBytes
    //       = (uint32_t)sbt.missProgRecordSize;
    //     localSBT.missRecordCount
    //       = (uint32_t)sbt.missProgRecordCount;

    //     assert("check hit records built" && sbt.hitGroupRecordCount != 0);
    //     localSBT.hitgroupRecordBase
    //       = (CUdeviceptr)sbt.hitGroupRecordsBuffer.get();
    //     localSBT.hitgroupRecordStrideInBytes
    //       = (uint32_t)sbt.hitGroupRecordSize;
    //     localSBT.hitgroupRecordCount
    //       = (uint32_t)sbt.hitGroupRecordCount;
    //   }

    //   OPTIX_CALL(Launch(context->pipeline,
    //                     lp->stream,
    //                     (CUdeviceptr)lp->deviceMemory.get(),
    //                     lp->deviceMemory.sizeInBytes,
    //                     &localSBT,
    //                     dims.x,dims.y,1
    //                     ));
    //   STACK_POP_ACTIVE();
    // }
    

  
} // ::owl

