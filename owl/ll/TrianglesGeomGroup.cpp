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
#include <fstream>

#define LOG(message)                                            \
  if (Context::logging()) \
    std::cout << "#owl.ll(" << context->owlDeviceID << "): "    \
  << message                                                    \
  << std::endl

#define LOG_OK(message)                                 \
  if (Context::logging()) \
    std::cout << OWL_TERMINAL_GREEN                     \
  << "#owl.ll(" << context->owlDeviceID << "): "        \
  << message << OWL_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  if (Context::logging()) \
    std::cout << "#owl.ll(" << owlDeviceID << "): "     \
  << message                                            \
  << std::endl

#define CLOG_OK(message)                                \
  if (Context::logging()) \
  std::cout << OWL_TERMINAL_GREEN                       \
  << "#owl.ll(" << owlDeviceID << "): "                 \
  << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    void Device::trianglesGeomGroupCreate(int groupID,
                                          const int *geomIDs,
                                          size_t geomCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] ==nullptr);
        
      TrianglesGeomGroup *group
        = new TrianglesGeomGroup(geomCount,
                                 sbt.rangeAllocator.alloc(geomCount));
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      // set children - todo: move to separate (API?) function(s)!?
      if (geomIDs) {
      for (int childID=0;childID<geomCount;childID++) {
        int geomID = geomIDs[childID];
        assert("check geom child geom ID is valid" && geomID >= 0);
        assert("check geom child geom ID is valid" && geomID <  geoms.size());
        Geom *geom = geoms[geomID];
        assert("check geom indexed child geom valid" && geom != nullptr);
        assert("check geom is valid type" && geom->primType() == TRIANGLES);
        geom->numTimesReferenced++;
        group->children[childID] = geom;
      }
      }
    }

    void TrianglesGeomGroup::destroyAccel(Context *context) 
    {
      context->pushActive();
      if (traversable) {
        bvhMemory.free();
        traversable = 0;
      }
      context->popActive();
    }
    
    void TrianglesGeomGroup::buildAccel(Context *context) 
    {
      assert("check does not yet exist" && traversable == 0);
      assert("check does not yet exist" && bvhMemory.empty());
      
      context->pushActive();
      LOG("building triangles accel over "
          << children.size() << " geometries");

      size_t sumPrims = 0;
      uint32_t maxPrimsPerGAS = 0;
      optixDeviceContextGetProperty
        (context->optixContext,
         OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
         &maxPrimsPerGAS,
         sizeof(maxPrimsPerGAS));
      
      // ==================================================================
      // create triangle inputs
      // ==================================================================
      //! the N build inputs that go into the builder
      std::vector<OptixBuildInput> triangleInputs(children.size());
      /*! *arrays* of the vertex pointers - the buildinputs cointina
       *pointers* to the pointers, so need a temp copy here */
      std::vector<CUdeviceptr> vertexPointers(children.size());
      std::vector<CUdeviceptr> indexPointers(children.size());

      // for now we use the same flags for all geoms
      uint32_t triangleInputFlags[1] = { 0 };
      // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

      // now go over all children to set up the buildinputs
      for (int childID=0;childID<children.size();childID++) {
        // the three fields we're setting:
        CUdeviceptr     &d_vertices    = vertexPointers[childID];
        CUdeviceptr     &d_indices     = indexPointers[childID];
        OptixBuildInput &triangleInput = triangleInputs[childID];

        // the child wer're setting them with (with sanity checks)
        Geom *geom = children[childID];
        assert("double-check geom isn't null" && geom != nullptr);
        assert("sanity check refcount" && geom->numTimesReferenced >= 0);
       
        TrianglesGeom *tris = dynamic_cast<TrianglesGeom*>(geom);
        assert("double-check it's really triangles" && tris != nullptr);
        
        // now fill in the values:
        d_vertices = (CUdeviceptr )tris->vertexPointer;
        if (d_vertices == 0)
          OWL_EXCEPT("in TrianglesGeomGroup::buildAccel(): "
                     "triangles geom has null vertex array");
        assert("triangles geom has vertex array set" && d_vertices);
        
        d_indices  = (CUdeviceptr )tris->indexPointer;
        assert("triangles geom has index array set" && d_indices);

        triangleInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        auto &ta = triangleInput.triangleArray;
        ta.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        ta.vertexStrideInBytes = (uint32_t)tris->vertexStride;
        ta.numVertices         = (uint32_t)tris->vertexCount;
        ta.vertexBuffers       = &d_vertices;
      
        ta.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        ta.indexStrideInBytes  = (uint32_t)tris->indexStride;
        ta.numIndexTriplets    = (uint32_t)tris->indexCount;
        ta.indexBuffer         = d_indices;

        // -------------------------------------------------------
        // sanity check that we don't have too many prims
        // -------------------------------------------------------
        sumPrims += ta.numIndexTriplets;
        if (sumPrims > maxPrimsPerGAS) 
          throw std::runtime_error("number of prim in user geom group exceeds "
                                   "OptiX's MAX_PRIMITIVES_PER_GAS limit");
        
        // we always have exactly one SBT entry per shape (i.e., triangle
        // mesh), and no per-primitive materials:
        ta.flags                       = triangleInputFlags;
        // iw, jan 7, 2020: note this is not the "actual" number of
        // SBT entires we'll generate when we build the SBT, only the
        // number of per-ray-type 'groups' of SBT entities (i.e., before
        // scaling by the SBT_STRIDE that gets passed to
        // optixTrace. So, for the build input this value remains *1*).
        ta.numSbtRecords               = 1; 
        ta.sbtIndexOffsetBuffer        = 0; 
        ta.sbtIndexOffsetSizeInBytes   = 0; 
        ta.sbtIndexOffsetStrideInBytes = 0; 
      }
      
      // ==================================================================
      // BLAS setup: buildinputs set up, build the blas
      // ==================================================================
      
      // ------------------------------------------------------------------
      // first: compute temp memory for bvh
      // ------------------------------------------------------------------
      OptixAccelBuildOptions accelOptions = {};
      accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        |
        OPTIX_BUILD_FLAG_ALLOW_COMPACTION;

      accelOptions.motionOptions.numKeys  = 1;
      accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
      
      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage
                  (context->optixContext,
                   &accelOptions,
                   triangleInputs.data(),
                   (uint32_t)triangleInputs.size(),
                   &blasBufferSizes
                   ));
      
      // ------------------------------------------------------------------
      // ... and allocate buffers: temp buffer, initial (uncompacted)
      // BVH buffer, and a one-single-size_t buffer to store the
      // compacted size in
      // ------------------------------------------------------------------

      // temp memory:
      DeviceMemory tempBuffer;
      tempBuffer.alloc(blasBufferSizes.tempSizeInBytes);

      // buffer for initial, uncompacted bvh
      DeviceMemory outputBuffer;
      outputBuffer.alloc(blasBufferSizes.outputSizeInBytes);

      // single size-t buffer to store compacted size in
      DeviceMemory compactedSizeBuffer;
      compactedSizeBuffer.alloc(sizeof(uint64_t));
      
      // ------------------------------------------------------------------
      // now execute initial, uncompacted build
      // ------------------------------------------------------------------
      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)compactedSizeBuffer.get();
      
      OPTIX_CHECK(optixAccelBuild(context->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  triangleInputs.data(),
                                  (uint32_t)triangleInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)outputBuffer.get(),
                                  outputBuffer.size(),
                                  /* the traversable we're building: */ 
                                  &traversable,
                                  /* we're also querying compacted size: */
                                  &emitDesc,1u
                                  ));
      CUDA_SYNC_CHECK();
      
      // ==================================================================
      // perform compaction
      // ==================================================================

      // download builder's compacted size from device
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize);

      // alloc the buffer...
      bvhMemory.alloc(compactedSize);
      // ... and perform compaction
      OPTIX_CALL(AccelCompact(context->optixContext,
                              /*TODO: stream:*/0,
                              // OPTIX_COPY_MODE_COMPACT,
                              traversable,
                              (CUdeviceptr)bvhMemory.get(),
                              bvhMemory.size(),
                              &traversable));
      CUDA_SYNC_CHECK();

#if 0
      std::vector<uint8_t> dumpBuffer(bvhMemory.size());
      bvhMemory.download(dumpBuffer.data());
      std::ofstream dump("/tmp/bvhMemory.bin",std::ios::binary);
      dump.write((char*)dumpBuffer.data(),dumpBuffer.size());
      PRINT(dumpBuffer.size());
#endif
      
      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
      tempBuffer.free();
      compactedSizeBuffer.free();
      
      context->popActive();

      LOG_OK("successfully build triangles geom group accel");
    }
    
  } // ::owl::ll
} //::owl
