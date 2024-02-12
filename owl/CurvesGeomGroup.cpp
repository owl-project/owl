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

#include "CurvesGeomGroup.h"
#include "CurvesGeom.h"
#include "Context.h"

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

  /*! pretty-printer, for printf-debugging */
  std::string CurvesGeomGroup::toString() const
  {
    return "CurvesGeomGroup";
  }
  
  /*! constructor - mostly passthrough to parent class */
  CurvesGeomGroup::CurvesGeomGroup(Context *const context,
                                         size_t numChildren,
                                         unsigned int _buildFlags)
    : GeomGroup(context,numChildren), 
      buildFlags(( (_buildFlags > 0) ? _buildFlags : defaultBuildFlags)
                 |
                 OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
                 )
      // buildFlags(OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS)
  {}
  
  void CurvesGeomGroup::updateMotionBounds()
  {
    // only need this for older version of optix that wouldn't even support curves
    OWL_NOTIMPLEMENTED;
  }
  
  void CurvesGeomGroup::buildAccel(LaunchParams::SP launchParams)
  {
    for (auto device : context->getDevices()) 
      buildAccelOn<true>(device);

    if (context->motionBlurEnabled)
      updateMotionBounds();
  }
  
  void CurvesGeomGroup::refitAccel(LaunchParams::SP launchParams)
  {
    for (auto device : context->getDevices()) 
      buildAccelOn<false>(device);
    
    if (context->motionBlurEnabled)
      updateMotionBounds();
  }
  
  template<bool FULL_REBUILD>
  void CurvesGeomGroup::buildAccelOn(const DeviceContext::SP &device) 
  {
// #if OPTIX_VERSION >= 70300
#if OWL_CAN_DO_CURVES
    DeviceData &dd = getDD(device);

    if (FULL_REBUILD && !dd.bvhMemory.empty())
      dd.bvhMemory.free();

    if (!FULL_REBUILD && dd.bvhMemory.empty())
      throw std::runtime_error("trying to refit an accel struct that has not been previously built");

    if (!FULL_REBUILD && !(buildFlags & OPTIX_BUILD_FLAG_ALLOW_UPDATE))
      throw std::runtime_error("trying to refit an accel struct that was not built with OPTIX_BUILD_FLAG_ALLOW_UPDATE");

    if (FULL_REBUILD) {
      dd.memFinal = 0;
      dd.memPeak = 0;
    }
   
    SetActiveGPU forLifeTime(device);
    LOG("building curves accel over "
        << geometries.size() << " geometries");
    size_t   sumPrims = 0;
    uint32_t maxPrimsPerGAS = 0;
    optixDeviceContextGetProperty
      (device->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
       &maxPrimsPerGAS,
       sizeof(maxPrimsPerGAS));

    assert(!geometries.empty());
    CurvesGeom::SP child0 = geometries[0]->as<CurvesGeom>();
    assert(child0);
    int numKeys = (int)child0->verticesBuffers.size();
    assert(numKeys > 0);
    const bool hasMotion = (numKeys > 1);
    if (hasMotion) assert(context->motionBlurEnabled);
    
    // ==================================================================
    // create curve inputs
    // ==================================================================
    //! the N build inputs that go into the builder
    std::vector<OptixBuildInput> buildInputs(geometries.size());
    // one build flag per build input
    // std::vector<uint32_t> buildInputFlags(geometries.size());

    // now go over all geometries to set up the buildinputs
    for (size_t childID=0;childID<geometries.size();childID++) {
      // the child wer're setting them with (with sanity checks)
      CurvesGeom::SP curves = geometries[childID]->as<CurvesGeom>();
      assert(curves);
      
      if (curves->verticesBuffers.size() != (size_t)numKeys)
        OWL_RAISE("invalid combination of meshes with "
                  "different motion keys in the same "
                  "curves geom group");
      CurvesGeom::DeviceData &curvesDD = curves->getDD(device);
      
      CUdeviceptr     *d_vertices       = curvesDD.verticesPointers.data();
      assert(d_vertices);
      CUdeviceptr     *d_widths         = curvesDD.widthsPointers.data();
      assert(d_widths);
      // CUdeviceptr     *d_segmentIndices = curvesDD.segmentIndicesPointers.data();
      // assert(d_widths);

      OptixBuildInput &buildInput = buildInputs[childID];

      buildInput = {}; // init defaults, whatever they might be
      
      buildInput.type = OPTIX_BUILD_INPUT_TYPE_CURVES;
      auto &curveArray = buildInput.curveArray;

      auto curvesGT = curves->geomType->as<CurvesGeomType>();//getTypeDD(device);
      
      switch(curvesGT->degree) {
      case 1:
        curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
        break;
      case 2:
        curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
        break;
      case 3:
        curveArray.curveType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
        break;
      default:
        OWL_RAISE("invalid/unsupported curve degree for owl curves geometry");
      }
      
      curveArray.numPrimitives        = curves->segmentIndicesCount;//1;
      curveArray.vertexBuffers        = d_vertices;//vertexBufferPointers;
      curveArray.numVertices          = curves->vertexCount;//static_cast<uint32_t>( vertices.size() );
      curveArray.vertexStrideInBytes  = sizeof(vec3f);
      curveArray.widthBuffers         = d_widths;//widthsBufferPointers;
      curveArray.widthStrideInBytes   = sizeof(float);
      curveArray.normalBuffers        = 0;
      curveArray.normalStrideInBytes  = 0;
      curveArray.indexBuffer          = curvesDD.indicesPointer;//d_segmentIndices;
      curveArray.indexStrideInBytes   = sizeof(int);
      curveArray.flag                 = OPTIX_GEOMETRY_FLAG_NONE;
      curveArray.primitiveIndexOffset = 0;
      curveArray.endcapFlags          = //OPTIX_CURVE_ENDCAP_DEFAULT;
        
      curvesGT->forceCaps
        ? OPTIX_CURVE_ENDCAP_ON
        : OPTIX_CURVE_ENDCAP_DEFAULT;

      // -------------------------------------------------------
      // sanity check that we don't have too many prims
      // -------------------------------------------------------
      sumPrims += curveArray.numPrimitives;
    }
    
    // -------------------------------------------------------
    // sanity check that we don't have too many prims
    // -------------------------------------------------------
    if (sumPrims > maxPrimsPerGAS) 
      OWL_RAISE("number of prim in user geom group exceeds "
                "OptiX's MAX_PRIMITIVES_PER_GAS limit");
    
    // ==================================================================
    // BLAS setup: buildinputs set up, build the blas
    // ==================================================================
      
    // ------------------------------------------------------------------
    // first: compute temp memory for bvh
    // ------------------------------------------------------------------
    OptixAccelBuildOptions accelOptions = {};

    if (numKeys > 1) {
      accelOptions.motionOptions.numKeys   = numKeys;
      accelOptions.motionOptions.flags     = 0;
      accelOptions.motionOptions.timeBegin = 0.f;
      accelOptions.motionOptions.timeEnd   = 1.f;
    }
    accelOptions.buildFlags
      =
      this->buildFlags
      // | OPTIX_BUILD_FLAG_ALLOW_COMPACTION
      // |
      // OPTIX_BUILD_FLAG_ALLOW_RANDOM_VERTEX_ACCESS
      ;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
      
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage
                (device->optixContext,
                 &accelOptions,
                 buildInputs.data(),
                 (uint32_t)buildInputs.size(),
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
        << prettyNumber(buildInputs.size()) << " curve geom groups, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");

    // temp memory:
    DeviceMemory tempBuffer;
    tempBuffer.alloc(FULL_REBUILD
                     ?blasBufferSizes.tempSizeInBytes
                     :blasBufferSizes.tempUpdateSizeInBytes);
    if (FULL_REBUILD) {
      // Only track this on first build, assuming temp buffer gets smaller for refit
      dd.memPeak += tempBuffer.size();
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
        dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
        dd.memPeak += dd.bvhMemory.size();
        dd.memFinal = dd.bvhMemory.size();
      }
    }

    // Build or refit

    if (FULL_REBUILD && allowCompaction) {

      compactedSizeBuffer.alloc(sizeof(uint64_t));
      dd.memPeak += compactedSizeBuffer.size();

      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)compactedSizeBuffer.get();

      // Initial, uncompacted build
      OPTIX_CHECK(optixAccelBuild(device->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  buildInputs.data(),
                                  (uint32_t)buildInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)outputBuffer.get(),
                                  outputBuffer.size(),
                                  /* the traversable we're building: */ 
                                  &dd.traversable,
                                  /* we're also querying compacted size: */
                                  &emitDesc,1u
                                  ));
    } else {

      // This is either a full rebuild operation _without_ compaction, or a refit.
      // The operation has already been stored in accelOptions.

      OPTIX_CHECK(optixAccelBuild(device->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  buildInputs.data(),
                                  (uint32_t)buildInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)dd.bvhMemory.get(),
                                  dd.bvhMemory.size(),
                                  /* the traversable we're building: */ 
                                  &dd.traversable,
                                  /* we're also querying compacted size: */
                                  nullptr,0
                                  ));
    }
    OWL_CUDA_SYNC_CHECK();
    
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
    OWL_CUDA_SYNC_CHECK();
      
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    if (FULL_REBUILD)
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
    tempBuffer.free();
    if (FULL_REBUILD)
      compactedSizeBuffer.free();
    
    LOG_OK("successfully build curves geom group accel");
#else
    throw std::runtime_error("This version of OWL was compiled with an OptiX version that does not yet support curves. Please re-build with a newer version of OptiX if you do want to use curves");
#endif
  }
  
} // ::owl
