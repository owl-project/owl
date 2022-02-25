// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "UserGeomGroup.h"
#include "Context.h"

#define FREE_EARLY 1

// useful for profiling with nvtxRangePushA("A"); and nvtxRangePop();
// #include "C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v11.6\include\nvtx3\nvToolsExt.h"

#define LOG(message)                                            \
  if (Context::logging())                                       \
    std::cout << "#owl(" << device->ID << "): "                 \
              << message                                        \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  
  UserGeomGroup::UserGeomGroup(Context *const context,
                               size_t numChildren,
                               unsigned int _buildFlags,
                               uint32_t _numKeys)
    : GeomGroup(context,numChildren), 
    buildFlags( (_buildFlags > 0) ? _buildFlags : defaultBuildFlags),
    numKeys(_numKeys)
  {}

  void UserGeomGroup::buildOrRefit(bool FULL_REBUILD)
  {
    for (auto child : geometries) {
      UserGeom::SP userGeom = child->as<UserGeom>();
      assert(userGeom);
      for (auto device : context->getDevices())
        userGeom->executeBoundsProgOnPrimitives(device);
    }
    
    for (auto device : context->getDevices())
      if (FULL_REBUILD)
        buildAccelOn<true>(device);
      else
        buildAccelOn<false>(device);
  }
  
  void UserGeomGroup::buildAccel()
  {
    buildOrRefit(true);
  }

  void UserGeomGroup::refitAccel()
  {
    buildOrRefit(false);
  }

  /*! low-level accel structure builder for given device */
  template<bool FULL_REBUILD>
  void UserGeomGroup::buildAccelOn(const DeviceContext::SP &device)
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;

    // if (FULL_REBUILD && !dd.bvhMemory.empty())
      // dd.bvhMemory.free(); 
      // NM: Don't do this, freeing is expensive. Instead, reuse previous allocation if possible.
      // Note, refitting isn't always an option, eg for photon maps that change each frame
    // cudaDeviceSynchronize();

    if (!FULL_REBUILD && dd.bvhMemory.empty())
      throw std::runtime_error("trying to refit an accel struct that has not been previously built");

    if (!FULL_REBUILD && !(buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE))
      throw std::runtime_error("trying to refit an accel struct that was not built with OPTIX_BUILD_FLAG_ALLOW_UPDATE");

    if (FULL_REBUILD) {
      dd.memFinal = 0;
      dd.memPeak = 0;
    }
      
    SetActiveGPU forLifeTime(device);
    LOG("building user accel over "
        << geometries.size() << " geometries");

    size_t sumPrims = 0;
    uint32_t maxPrimsPerGAS = 0;
    optixDeviceContextGetProperty
      (device->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
       &maxPrimsPerGAS,
       sizeof(maxPrimsPerGAS));
    
    // ==================================================================
    // create user geom inputs
    // ==================================================================
    //! the N build inputs that go into the builder
    std::vector<OptixBuildInput> userGeomInputs(geometries.size());
    /*! *arrays* of the vertex pointers - the buildinputs contain
     *pointers* to the pointers, so need a temp copy here */
    std::vector<std::vector<CUdeviceptr>> boundsPointers(numKeys);
    for (uint32_t i = 0; i < numKeys; ++i) {
      boundsPointers[i] = std::vector<CUdeviceptr>(geometries.size());
    }

    // for now we use the same flags for all geoms
    std::vector<uint32_t> userGeomInputFlags(geometries.size());
    std::vector<std::vector<CUdeviceptr>> d_boundsList;

    // now go over all geometries to set up the buildinputs
    for (size_t childID=0;childID<geometries.size();childID++) {
      // the three fields we're setting:

      UserGeom::SP child = geometries[childID]->as<UserGeom>();
      assert(child);

      sumPrims += child->primCount;
      if (sumPrims > maxPrimsPerGAS) 
        OWL_RAISE("number of prim in user geom group exceeds "
                  "OptiX's MAX_PRIMITIVES_PER_GAS limit");

      UserGeom::DeviceData &ugDD = child->getDD(device);

      OptixBuildInput &userGeomInput = userGeomInputs[childID];

      assert("user geom has enough motion keys" && ugDD.internalBufferForBoundsProgram.size() == numKeys);
      for (uint32_t i = 0; i < numKeys; ++i) {
        assert("user geom has valid bounds buffer" && ugDD.internalBufferForBoundsProgram[i].alloced());
        boundsPointers[i][childID] = (CUdeviceptr)ugDD.internalBufferForBoundsProgram[i].get();
      }

      userGeomInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
#if OPTIX_VERSION >= 70100
      auto &aa = userGeomInput.customPrimitiveArray;
#else
      auto &aa = userGeomInput.aabbArray;
#endif
      
      std::vector<CUdeviceptr> bounds;
      for (uint32_t i = 0; i < numKeys; ++i) {
        bounds.push_back(boundsPointers[i][childID]);
      }
      d_boundsList.push_back(bounds);
      aa.aabbBuffers   = (CUdeviceptr*)d_boundsList[childID].data();
      aa.numPrimitives = (uint32_t)child->primCount;
      aa.strideInBytes = sizeof(box3f);
      aa.primitiveIndexOffset = 0;
      
      // we always have exactly one SBT entry per shape (i.e., triangle
      // mesh), and no per-primitive materials:
      userGeomInputFlags[childID]    = 0;
      aa.flags                       = &userGeomInputFlags[childID];
      // iw, jan 7, 2020: note this is not the "actual" number of
      // SBT entires we'll generate when we build the SBT, only the
      // number of per-ray-type 'groups' of SBT enties (i.e., before
      // scaling by the SBT_STRIDE that gets passed to
      // optixTrace. So, for the build input this value remains *1*).
      aa.numSbtRecords               = 1; 
      aa.sbtIndexOffsetBuffer        = 0; 
      aa.sbtIndexOffsetSizeInBytes   = 0; 
      aa.sbtIndexOffsetStrideInBytes = 0; 
    }

    // ==================================================================
    // BLAS setup: buildinputs set up, build the blas
    // ==================================================================
      
    // ------------------------------------------------------------------
    // first: compute temp memory for bvh
    // ------------------------------------------------------------------
    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = this->buildFlags;
    accelOptions.motionOptions.numKeys  = numKeys;
    accelOptions.motionOptions.timeBegin = 0.f;
    accelOptions.motionOptions.timeEnd   = 1.f;
    accelOptions.motionOptions.flags = OPTIX_MOTION_FLAG_START_VANISH;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
    
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (optixContext,
                 &accelOptions,
                 userGeomInputs.data(),
                 (uint32_t)userGeomInputs.size(),
                 &blasBufferSizes
                 ));
   
    // ------------------------------------------------------------------
    // ... and allocate buffers: temp buffer, initial (uncompacted)
    // BVH buffer, and a one-single-size_t buffer to store the
    // compacted size in
    // ------------------------------------------------------------------
      
    const size_t tempSize
        = FULL_REBUILD
        ? blasBufferSizes.tempSizeInBytes
        : blasBufferSizes.tempUpdateSizeInBytes;
    LOG("starting to build/refit "
        << prettyNumber(userGeomInputs.size()) << " user geoms, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");

    // temp memory:
    size_t sizeInBytes = (FULL_REBUILD
       ? blasBufferSizes.tempSizeInBytes
       : blasBufferSizes.tempUpdateSizeInBytes);

    if (dd.tempBuffer.sizeInBytes < sizeInBytes)
    {
      if (!dd.tempBuffer.empty()) dd.tempBuffer.free();
      dd.tempBuffer.alloc(sizeInBytes);      
    }      

    if (FULL_REBUILD) {
      dd.memPeak += dd.tempBuffer.size();
    }

    const bool allowCompaction = (buildFlags & OPTIX_BUILD_FLAG_ALLOW_COMPACTION);

    // Optional buffers only used when compaction is allowed
    DeviceMemory outputBuffer;
    DeviceMemory compactedSizeBuffer;

    // Allocate output buffer for initial build
    if (FULL_REBUILD) {
      if (allowCompaction) {
        outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);
        dd.memPeak += outputBuffer.size();
      } else {
        if (dd.bvhMemory.sizeInBytes < blasBufferSizes.outputSizeInBytes) {
          if (!dd.bvhMemory.empty()) { 
            dd.bvhMemory.free(); 
          }
          dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
          dd.memPeak += dd.bvhMemory.size();
          dd.memFinal = dd.bvhMemory.size();
        }
      }
    }

    // Build or refit

    if (FULL_REBUILD && allowCompaction) {

      compactedSizeBuffer.alloc(sizeof(uint64_t));
      dd.memPeak += compactedSizeBuffer.size();

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)compactedSizeBuffer.get();

      OPTIX_CHECK(optixAccelBuild(optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  userGeomInputs.data(),
                                  (uint32_t)userGeomInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)dd.tempBuffer.get(),
                                  dd.tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)outputBuffer.get(),
                                  outputBuffer.size(),
                                  /* the dd.traversable we're building: */ 
                                  &dd.traversable,
                                  /* we're also querying compacted size: */
                                  &emitDesc,1u
                                  ));
    } else {
      OPTIX_CHECK(optixAccelBuild(optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  userGeomInputs.data(),
                                  (uint32_t)userGeomInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)dd.tempBuffer.get(),
                                  dd.tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)dd.bvhMemory.get(),
                                  dd.bvhMemory.size(),
                                  /* the dd.traversable we're building: */ 
                                  &dd.traversable,
                                  /* not querying anything */
                                  nullptr,0
                                  ));
    }
 
    // ==================================================================
    // perform compaction
    // ==================================================================
    
    if (FULL_REBUILD && allowCompaction) {
      // download builder's compacted size from device
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize);
      
      dd.bvhMemory.alloc(compactedSize);
      // ... and perform compaction
      OPTIX_CALL(AccelCompact(device->optixContext,
                              /*TODO: stream:*/0,
                              // OPTIX_COPY_MODE_COMPACT,
                              dd.traversable,
                              (CUdeviceptr)dd.bvhMemory.get(),
                              dd.bvhMemory.size(),
                              &dd.traversable));
      dd.memPeak += dd.bvhMemory.size();
      dd.memFinal = dd.bvhMemory.size();
    }
 
    // ==================================================================
    // finish - clean up
    // ==================================================================
    LOG_OK("successfully built user geom group accel");

    #if FREE_EARLY == 1
    if (dd.tempBuffer.alloced())
      dd.tempBuffer.free();
    #endif

    // size_t sumPrims = 0;
    size_t sumBoundsMem = 0;
    for (size_t childID=0;childID<geometries.size();childID++) {
      UserGeom::SP child = geometries[childID]->as<UserGeom>();
      assert(child);
      
      UserGeom::DeviceData &ugDD = child->getDD(device);
      
      for (uint32_t i = 0; i < numKeys; ++i) {
        sumBoundsMem += ugDD.internalBufferForBoundsProgram[i].sizeInBytes;

        // don't do this unless absolutely necessary. For performance reasons, recycle this memory across builds if possible.
        #if FREE_EARLY == 1
        if (ugDD.internalBufferForBoundsProgram[i].alloced())
          ugDD.internalBufferForBoundsProgram[i].free();
        #endif
      }
    }
    if (FULL_REBUILD)
      dd.memPeak += sumBoundsMem;
  }
    
} // ::owl
