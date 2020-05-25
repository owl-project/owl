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

#include "Device.h"
#include <fstream>

#define LOG(message)                                            \
  if (Context::logging())                                       \
    std::cout << "#owl.ll(" << context->owlDeviceID << "): "    \
              << message                                        \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl.ll(" << context->owlDeviceID << "): "    \
              << message << OWL_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  if (Context::logging())                               \
    std::cout << "#owl.ll(" << owlDeviceID << "): "     \
              << message                                \
              << std::endl

#define CLOG_OK(message)                                        \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl.ll(" << owlDeviceID << "): "             \
              << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    void dbgPrintBVHSizes(size_t numItems,
                          size_t boundsArraySize,
                          size_t tempMemSize,
                          size_t initialBVHSize,
                          size_t finalBVHSize)
    {
#if _WIN32
      static char *wantToPrint = nullptr;
#else
      static char *wantToPrint = getenv("OWL_PRINT_BVH_MEM");
#endif
      if (!wantToPrint) return;
      
      std::cout << OWL_TERMINAL_YELLOW
                << "@owl: bvh build mem: "
                << prettyNumber(numItems) << " items, "
                << prettyNumber(boundsArraySize) << "b bounds, "
                << prettyNumber(tempMemSize) << "b temp, "
                << prettyNumber(initialBVHSize) << "b initBVH, "
                << prettyNumber(finalBVHSize) << "b finalBVH"
                << OWL_TERMINAL_DEFAULT
                << std::endl;
    }
    
    void Device::groupBuildPrimitiveBounds(int groupID,
                                           size_t maxGeomDataSize,
                                           LLOWriteUserGeomBoundsDataCB cb,
                                           const void *cbData)
    {
      context->pushActive();
      UserGeomGroup *ugg
        = checkGetUserGeomGroup(groupID);
      
      std::vector<uint8_t> userGeomData(maxGeomDataSize);
      DeviceMemory tempMem;
      tempMem.alloc(maxGeomDataSize);
      size_t sumPrims = 0;
      uint32_t maxPrimsPerGAS = 0;
      optixDeviceContextGetProperty
        (context->optixContext,
         OPTIX_DEVICE_PROPERTY_LIMIT_MAX_PRIMITIVES_PER_GAS,
         &maxPrimsPerGAS,
         sizeof(maxPrimsPerGAS));
      
      for (int childID=0;childID<ugg->children.size();childID++) {
        Geom *child = ugg->children[childID];
        assert("double-check valid child geom" && child != nullptr);
        assert(child);
        UserGeom *ug = (UserGeom *)child;
        ug->internalBufferForBoundsProgram.alloc(ug->numPrims*sizeof(box3f));
        ug->d_boundsMemory = ug->internalBufferForBoundsProgram.get();
        
        if (childID < 10)
          LOG("calling user geom callback to set up user geometry bounds call data");
        else if (childID == 10)
          LOG("(more instances may follow)");
          
        cb(userGeomData.data(),context->owlDeviceID,
           ug->geomID,childID,cbData); 
        
        uint32_t numPrims = (uint32_t)ug->numPrims;
        sumPrims += numPrims;
        if (sumPrims > maxPrimsPerGAS) 
          throw std::runtime_error("number of prim in user geom group exceeds "
                                   "OptiX's MAX_PRIMITIVES_PER_GAS limit");
        // size of each thread block during bounds function call
        vec3i blockDims(32,32,1);
        uint32_t threadsPerBlock = blockDims.x*blockDims.y*blockDims.z;
        
        uint32_t numBlocks = owl::common::divRoundUp(numPrims,threadsPerBlock);
        uint32_t numBlocks_x
          = 1+uint32_t(powf((float)numBlocks,1.f/3.f));
        uint32_t numBlocks_y
          = 1+uint32_t(sqrtf((float)(numBlocks/numBlocks_x)));
        uint32_t numBlocks_z
          = owl::common::divRoundUp(numBlocks,numBlocks_x*numBlocks_y);
        
        vec3i gridDims(numBlocks_x,numBlocks_y,numBlocks_z);
        
        tempMem.upload(userGeomData);
        
        void  *d_geomData = tempMem.get();//nullptr;
        vec3f *d_boundsArray = (vec3f*)ug->d_boundsMemory;
        void  *args[] = {
          &d_geomData,
          &d_boundsArray,
          (void *)&numPrims
        };
        
        GeomType *gt = checkGetGeomType(ug->geomTypeID);
        CUstream stream = context->stream;
        if (!gt->boundsFuncKernel)
          throw std::runtime_error("bounds kernel set, but not yet compiled - did you forget to call BuildPrograms() before (User)GroupAccelBuild()!?");
        
        CUresult rc
          = cuLaunchKernel(gt->boundsFuncKernel,
                           gridDims.x,gridDims.y,gridDims.z,
                           blockDims.x,blockDims.y,blockDims.z,
                           0, stream, args, 0);
        if (rc) {
          const char *errName = 0;
          cuGetErrorName(rc,&errName);
          PRINT(errName);
          exit(0);
        }
      }
      tempMem.free();
      cudaDeviceSynchronize();
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
      assert("check does not yet exist" && bvhMemory.empty());
      
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
      uint32_t userGeomInputFlags[1]
        = { 0 };
      // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };

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
               && (userGeom->internalBufferForBoundsProgram.alloced()
                   || userGeom->d_boundsMemory));
        d_bounds = (CUdeviceptr)userGeom->d_boundsMemory;
        
        userGeomInput.type = OPTIX_BUILD_INPUT_TYPE_CUSTOM_PRIMITIVES;
        auto &aa = userGeomInput.aabbArray;
        aa.aabbBuffers   = &d_bounds;
        aa.numPrimitives = (uint32_t)userGeom->numPrims;
        aa.strideInBytes = sizeof(box3f);
        aa.primitiveIndexOffset = 0;
      
        // we always have exactly one SBT entry per shape (i.e., triangle
        // mesh), and no per-primitive materials:
        aa.flags                       = userGeomInputFlags;
        // iw, jan 7, 2020: note this is not the "actual" number of
        // SBT entires we'll generate when we build the SBT, only the
        // number of per-ray-type 'groups' of SBT enties (i.e., before
        // scaling by the SBT_STRIDE that gets passed to
        // optixTrace. So, for the build input this value remains *1*).
        aa.numSbtRecords               = 1; //context->numRayTypes;
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
      accelOptions.buildFlags             =
        // OPTIX_BUILD_FLAG_PREFER_FAST_BUILD
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        ;
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

      bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
      OPTIX_CHECK(optixAccelBuild(context->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  userGeomInputs.data(),
                                  (uint32_t)userGeomInputs.size(),
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)bvhMemory.get(),
                                  bvhMemory.size(),
                                  /* the traversable we're building: */ 
                                  &traversable,
                                  /* we're also querying compacted size: */
                                  nullptr,0u
                                  ));

      CUDA_SYNC_CHECK();

#if 0
      // for debugging only - dumps the BVH to disk
      std::vector<uint8_t> dumpBuffer(outputBuffer.size());
      outputBuffer.download(dumpBuffer.data());
      std::ofstream dump("/tmp/outputBuffer.bin",std::ios::binary);
      dump.write((char*)dumpBuffer.data(),dumpBuffer.size());
      PRINT(dumpBuffer.size());
      exit(0);
#endif
      
      // ==================================================================
      // finish - clean up
      // ==================================================================

      tempBuffer.free();
      context->popActive();

      LOG_OK("successfully built user geom group accel");

      size_t sumPrims = 0;
      size_t sumBoundsMem = 0;
      for (int childID=0;childID<children.size();childID++) {
        Geom *geom = children[childID];
        UserGeom *userGeom = dynamic_cast<UserGeom*>(geom);
        sumPrims += userGeom->numPrims;
        sumBoundsMem += userGeom->internalBufferForBoundsProgram.sizeInBytes;
        if (userGeom->internalBufferForBoundsProgram.alloced())
          userGeom->internalBufferForBoundsProgram.free();
      }
      dbgPrintBVHSizes(/*numItems*/
                       sumPrims,
                       /*boundsArraySize*/
                       sumBoundsMem,
                       /*tempMemSize*/
                       blasBufferSizes.tempSizeInBytes,
                       /*initialBVHSize*/
                       blasBufferSizes.outputSizeInBytes,
                       /*finalBVHSize*/
                       0
                       );
    }
    

    void Device::userGeomGroupCreate(int groupID,
                                     const int *geomIDs,
                                     size_t childCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] == nullptr);

      UserGeomGroup *group
        = new UserGeomGroup(childCount,
                            sbt.rangeAllocator.alloc(childCount));
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      // set children - todo: move to separate (API?) function(s)!?
      if (geomIDs) {
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
    }

  } // ::owl::ll
} //::owl
