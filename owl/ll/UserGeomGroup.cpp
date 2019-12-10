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

#define LOG(message)                                            \
  std::cout << "#owl.ll(" << context->owlDeviceID << "): "      \
  << message                                                    \
  << std::endl

#define LOG_OK(message)                                 \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << context->owlDeviceID << "): "        \
  << message << GDT_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  std::cout << "#owl.ll(" << owlDeviceID << "): "       \
  << message                                            \
  << std::endl

#define CLOG_OK(message)                                \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << owlDeviceID << "): "                 \
  << message << GDT_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    void Device::groupBuildPrimitiveBounds(int groupID,
                                           size_t maxGeomDataSize,
                                           WriteUserGeomBoundsDataCB cb,
                                           void *cbData)
    {
      context->pushActive();
      UserGeomGroup *ugg
        = checkGetUserGeomGroup(groupID);
      
      std::vector<uint8_t> userGeomData(maxGeomDataSize);
      DeviceMemory tempMem;
      tempMem.alloc(maxGeomDataSize);
      for (int childID=0;childID<ugg->children.size();childID++) {
        Geom *child = ugg->children[childID];
        assert("double-check valid child geom" && child != nullptr);
        assert(child);
        UserGeom *ug = (UserGeom *)child;
        ug->internalBufferForBoundsProgram.alloc(ug->numPrims);
        ug->d_boundsMemory = ug->internalBufferForBoundsProgram.get();

        LOG("calling user geom callback to set up user geometry bounds call data");
        cb(userGeomData.data(),context->owlDeviceID,
           ug->geomID,childID,cbData); 

        // size of each thread block during bounds function call
		uint32_t boundsFuncBlockSize = 128;
		uint32_t numPrims = (uint32_t)ug->numPrims;
        vec3i blockDims(gdt::divRoundUp(numPrims,boundsFuncBlockSize),1,1);
        vec3i gridDims(boundsFuncBlockSize,1,1);

        tempMem.upload(userGeomData);
        
        void  *d_geomData = tempMem.get();//nullptr;
        vec3f *d_boundsArray = (vec3f*)ug->d_boundsMemory;
        void  *args[] = {
          &d_geomData,
          &d_boundsArray,
          (void *)&numPrims
        };

        DeviceMemory tempMem;
        GeomType *gt = checkGetGeomType(ug->geomTypeID);
        CUresult rc
          = cuLaunchKernel(gt->boundsFuncKernel,
                           blockDims.x,blockDims.y,blockDims.z,
                           gridDims.x,gridDims.y,gridDims.z,
                           0, 0, args, 0);
        if (rc) {
          const char *errName = 0;
          cuGetErrorName(rc,&errName);
          PRINT(errName);
          exit(0);
        }
        cudaDeviceSynchronize();
      }
      tempMem.free();
      context->popActive();
    }

    void UserGeomGroup::destroyAccel(Context *context) 
    {
      context->pushActive();
      if (traversable) {
        bvhMemory.free();
        traversable = 0;
      }
      context->popActive();
    }
    
    void UserGeomGroup::buildAccel(Context *context) 
    {
      assert("check does not yet exist" && traversable == 0);
      assert("check does not yet exist" && !bvhMemory.valid());
      
      context->pushActive();
      LOG("building user accel over "
          << children.size() << " geometries");
      // ==================================================================
      // create triangle inputs
      // ==================================================================
      //! the N build inputs that go into the builder
      std::vector<OptixBuildInput> userGeomInputs(children.size());
      /*! *arrays* of the vertex pointers - the buildinputs cointina
       *pointers* to the pointers, so need a temp copy here */
      std::vector<CUdeviceptr> boundsPointers(children.size());

     // for now we use the same flags for all geoms
      uint32_t userGeomInputFlags[1] = { 0 };
      // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

      // now go over all children to set up the buildinputs
      for (int childID=0;childID<children.size();childID++) {
        // the three fields we're setting:

        CUdeviceptr     &d_bounds = boundsPointers[childID];
        OptixBuildInput &userGeomInput = userGeomInputs[childID];
        
        // the child wer're setting them with (with sanity checks)
        Geom *geom = children[childID];
        assert("double-check geom isn't null" && geom != nullptr);
        assert("sanity check refcount" && geom->numTimesReferenced >= 0);
       
        UserGeom *userGeom = dynamic_cast<UserGeom*>(geom);
        assert("double-check it's really user"
               && userGeom != nullptr);
        assert("user geom has valid bounds buffer *or* user-supplied bounds"
               && (userGeom->internalBufferForBoundsProgram.valid()
                   || userGeom->d_boundsMemory));
        d_bounds = (CUdeviceptr)userGeom->d_boundsMemory;
        
        userGeomInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        auto &aa = userGeomInput.aabbArray;
        aa.aabbBuffers   = &d_bounds;
        aa.numPrimitives = (uint32_t)userGeom->numPrims;
        aa.strideInBytes = sizeof(box3f);
        aa.primitiveIndexOffset = 0;
      
        // we always have exactly one SBT entry per shape (ie, triangle
        // mesh), and no per-primitive materials:
        aa.flags                       = userGeomInputFlags;
        aa.numSbtRecords               = context->numRayTypes;
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
      accelOptions.buildFlags             = OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
      accelOptions.motionOptions.numKeys  = 1;
      accelOptions.operation              = OPTIX_BUILD_OPERATION_BUILD;
      
      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage
                  (context->optixContext,
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
                                  userGeomInputs.data(),
                                  (uint32_t)userGeomInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  (uint32_t)tempBuffer.size(),
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
      
      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
      tempBuffer.free();
      compactedSizeBuffer.free();
      
      context->popActive();

      LOG_OK("successfully build user geom group accel");
    }
    

    void Device::userGeomGroupCreate(int groupID,
                                     int *geomIDs,
                                     int childCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] == nullptr);
        
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && childCount == 0) ||
              (geomIDs != nullptr && childCount >  0)));
        
      UserGeomGroup *group
        = new UserGeomGroup(childCount,
                            sbt.rangeAllocator.alloc(childCount));
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      assert("currently have to specify all children at creation time" &&
             geomIDs != nullptr);
      // set children - todo: move to separate (api?) function(s)!?
      for (int childID=0;childID<childCount;childID++) {
        int geomID = geomIDs[childID];
        assert("check geom child geom ID is valid" && geomID >= 0);
        assert("check geom child geom ID is valid" && geomID <  geoms.size());
        Geom *geom = geoms[geomID];
        assert("check geom indexed child geom valid" && geom != nullptr);
        assert("check geom is valid type" && geom->primType() == USER);
        geom->numTimesReferenced++;
        group->children[childID] = geom;
      }
    }

  } // ::owl::ll
} //::owl
