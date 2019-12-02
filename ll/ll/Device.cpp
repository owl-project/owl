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
      CLOG("trying to create owl device on CUDA device #" << cudaDeviceID);
      
      cudaDeviceProp prop;
      cudaGetDeviceProperties(&prop, cudaDeviceID);
      CLOG(" - device: " << prop.name);

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
      CLOG("destroying owl device #"
           << owlDeviceID
           << " on CUDA device #" 
           << cudaDeviceID);
    }

    
    
    /*! construct a new owl device on given cuda device. throws an
      exception if for any reason that cannot be done */
    Device::Device(int owlDeviceID, int cudaDeviceID)
      : context(new Context(owlDeviceID,cudaDeviceID))
    {
      LOG_OK("successfully created owl device #" << owlDeviceID
             << " on CUDA device #" << cudaDeviceID);
    }
    

    Device::~Device()
    {
      destroyPipeline();
      
      modules.destroyOptixHandles(context);
      const int owlDeviceID = context->owlDeviceID;

      LOG("deleting context");
      delete context;
      context = nullptr;

      // iw - use CLOG here - regular LOG() will try to access
      // context, which by now is no longer available
      CLOG_OK("successfully destroyed owl device ...");
    }

    void Context::destroyPipeline()
    {
      if (pipeline) {
        // pushActive();
        OPTIX_CHECK(optixPipelineDestroy(pipeline));
        pipeline = nullptr;
        // popActive();
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
    
    void Device::setRayGen(int pgID,
                           int moduleID,
                           const char *progName)
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
    
    void Device::setMissProg(int pgID,
                             int moduleID,
                             const char *progName)
    {
      assert(pgID >= 0);
      assert(pgID < missProgPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      missProgPGs[pgID].program.moduleID = moduleID;
      missProgPGs[pgID].program.progName = progName;
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
      LOG("building " << modules.size() << " modules");
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      
      for (int moduleID=0;moduleID<modules.size();moduleID++) {
        Module &module = modules[moduleID];
        assert(module.module == nullptr);
        OPTIX_CHECK_LOG(optixModuleCreateFromPTX(context->optixContext,
                                                 &context->moduleCompileOptions,
                                                 &context->pipelineCompileOptions,
                                                 module.ptxCode,
                                                 strlen(module.ptxCode),
                                                 log,      // Log string
                                                 &sizeof_log,// Log string sizse
                                                 &module.module
                                                 ));
        assert(module.module != nullptr);
        LOG_OK("created module #" << moduleID);
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
      OptixProgramGroupDesc    pgDesc    = {};

      // ------------------------------------------------------------------
      // rayGen programs
      // ------------------------------------------------------------------
      for (int pgID=0;pgID<rayGenPGs.size();pgID++) {
        RayGenPG &pg     = rayGenPGs[pgID];
        Module   *module = modules.get(pg.program.moduleID);
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        std::string annotatedProgName
          = pg.program.progName
          ? std::string("__raygen__")+pg.program.progName
          : "";
        if (module) {
          assert(module->module != nullptr);
          assert(pg.program.progName != nullptr);
          pgDesc.raygen.module            = module->module;
          pgDesc.raygen.entryFunctionName = annotatedProgName.c_str();
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
      
      // ------------------------------------------------------------------
      // miss programs
      // ------------------------------------------------------------------
      for (int pgID=0;pgID<missProgPGs.size();pgID++) {
        MissProgPG &pg     = missProgPGs[pgID];
        Module *module = modules.get(pg.program.moduleID);
        pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_MISS;
        std::string annotatedProgName
          = pg.program.progName
          ? std::string("__miss__")+pg.program.progName
          : "";
        if (module) {
          assert(module->module != nullptr);
          assert(pg.program.progName != nullptr);
          pgDesc.miss.module            = module->module;
          pgDesc.miss.entryFunctionName = annotatedProgName.c_str();
        } else {
          pgDesc.miss.module            = nullptr;
          pgDesc.miss.entryFunctionName = nullptr;
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
      
      // ------------------------------------------------------------------
      // hitGroup programs
      // ------------------------------------------------------------------
      for (int pgID=0;pgID<hitGroupPGs.size();pgID++) {
        HitGroupPG &pg   = hitGroupPGs[pgID];
        pgDesc.kind      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;

        // ----------- closest hit -----------
        Module *moduleCH = modules.get(pg.closestHit.moduleID);
        std::string annotatedProgNameCH
          = pg.closestHit.progName
          ? std::string("__closesthit__")+pg.closestHit.progName
          : "";
        if (moduleCH) {
          assert(moduleCH->module != nullptr);
          assert(pg.closestHit.progName != nullptr);
          pgDesc.hitgroup.moduleCH            = moduleCH->module;
          pgDesc.hitgroup.entryFunctionNameCH = annotatedProgNameCH.c_str();
        } else {
          pgDesc.hitgroup.moduleCH            = nullptr;
          pgDesc.hitgroup.entryFunctionNameCH = nullptr;
        }
        // ----------- any hit -----------
        Module *moduleAH = modules.get(pg.anyHit.moduleID);
        std::string annotatedProgNameAH
          = pg.anyHit.progName
          ? std::string("__anyhit__")+pg.anyHit.progName
          : "";
        if (moduleAH) {
          assert(moduleAH->module != nullptr);
          assert(pg.anyHit.progName != nullptr);
          pgDesc.hitgroup.moduleAH            = moduleAH->module;
          pgDesc.hitgroup.entryFunctionNameAH = annotatedProgNameAH.c_str();
        } else {
          pgDesc.hitgroup.moduleAH            = nullptr;
          pgDesc.hitgroup.entryFunctionNameAH = nullptr;
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
      // ---------------------- rayGen ----------------------
      for (auto &pg : rayGenPGs) {
        if (pg.pg) optixProgramGroupDestroy(pg.pg);
        pg.pg = nullptr;
      }
      // ---------------------- hitGroup ----------------------
      for (auto &pg : hitGroupPGs) {
        if (pg.pg) optixProgramGroupDestroy(pg.pg);
        pg.pg = nullptr;
      }
      // ---------------------- miss ----------------------
      for (auto &pg : missProgPGs) {
        if (pg.pg) optixProgramGroupDestroy(pg.pg);
        pg.pg = nullptr;
      }
    }

    void Device::allocHitGroupPGs(size_t count)
    {
      assert(hitGroupPGs.empty());
      hitGroupPGs.resize(count);
    }
    
    void Device::allocRayGens(size_t count)
    {
      assert(rayGenPGs.empty());
      rayGenPGs.resize(count);
    }
    
    void Device::allocMissProgs(size_t count)
    {
      assert(missProgPGs.empty());
      missProgPGs.resize(count);
    }
      

    void Context::createPipeline(Device *device)
    {
      if (pipeline != nullptr)
        throw std::runtime_error("pipeline already created!?");
      
      std::vector<OptixProgramGroup> allPGs;
      assert(!device->rayGenPGs.empty());
      for (auto &pg : device->rayGenPGs)
        allPGs.push_back(pg.pg);
      assert(!device->hitGroupPGs.empty());
      for (auto &pg : device->hitGroupPGs)
        allPGs.push_back(pg.pg);
      assert(!device->missProgPGs.empty());
      for (auto &pg : device->missProgPGs)
        allPGs.push_back(pg.pg);

      if (allPGs.empty())
        throw std::runtime_error("trying to create a pipeline w/ 0 programs!?");
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      
      OPTIX_CHECK(optixPipelineCreate(optixContext,
                                      &pipelineCompileOptions,
                                      &pipelineLinkOptions,
                                      allPGs.data(),
                                      (uint32_t)allPGs.size(),
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
      


    
    void Device::createDeviceBuffer(int bufferID,
                                    size_t elementCount,
                                    size_t elementSize,
                                    const void *initData)
    {
      assert("check valid buffer ID" && bufferID >= 0);
      assert("check valid buffer ID" && bufferID <  buffers.size());
      assert("check buffer ID available" && buffers[bufferID] == nullptr);
      context->pushActive();
      DeviceBuffer *buffer = new DeviceBuffer(elementCount,elementSize);
      if (initData) {
        buffer->devMem.upload(initData,"createDeviceBuffer: uploading initData");
        LOG("uploading " << elementCount
            << " items of size " << elementSize
            << " from host ptr " << initData
            << " to device ptr " << buffer->devMem.get());
      }
      assert("check buffer properly created" && buffer != nullptr);
      buffers[bufferID] = buffer;
      context->popActive();
    }
    
    void Device::createHostPinnedBuffer(int bufferID,
                                        size_t elementCount,
                                        size_t elementSize,
                                        HostPinnedMemory::SP pinnedMem)
    {
      assert("check valid buffer ID" && bufferID >= 0);
      assert("check valid buffer ID" && bufferID <  buffers.size());
      assert("check buffer ID available" && buffers[bufferID] == nullptr);
      context->pushActive();
      Buffer *buffer = new HostPinnedBuffer(elementCount,elementSize,pinnedMem);
      assert("check buffer properly created" && buffer != nullptr);
      buffers[bufferID] = buffer;
      context->popActive();
    }
    
    void Device::trianglesGeomSetVertexBuffer(int geomID,
                                              int bufferID,
                                              int count,
                                              int stride,
                                              int offset)
    {
      TrianglesGeom *triangles
        = checkGetTrianglesGeom(geomID);
      assert("double-check valid geom" && triangles);
      
      Buffer   *buffer
        = checkGetBuffer(bufferID);
      assert("double-check valid buffer" && buffer);

      triangles->vertexPointer = addPointerOffset(buffer->get(),offset);
      triangles->vertexStride  = stride;
      triangles->vertexCount   = count;
    }

    /*! returns the given buffers device pointer */
    void *Device::bufferGetPointer(int bufferID)
    {
      return (void*)checkGetBuffer(bufferID)->d_pointer;
    }
      
    
    
    void Device::trianglesGeomSetIndexBuffer(int geomID,
                                             int bufferID,
                                             int count,
                                             int stride,
                                             int offset)
    {
      TrianglesGeom *triangles
        = checkGetTrianglesGeom(geomID);
      assert("double-check valid geom" && triangles);
      
      Buffer   *buffer
        = checkGetBuffer(bufferID);
      assert("double-check valid buffer" && buffer);

      triangles->indexPointer = addPointerOffset(buffer->get(),offset);
      triangles->indexCount   = count;
      triangles->indexStride  = stride;
    }
    
    void Device::groupBuildAccel(int groupID)
    {
      Group *group = checkGetGroup(groupID);
      group->destroyAccel(context);
      group->buildAccel(context);
    }

    /*! return given group's current traversable. note this function
      will *not* check if the group has alreadybeen built, if it
      has to be rebuilt, etc. */
    OptixTraversableHandle Device::groupGetTraversable(int groupID)
    {
      return checkGetGroup(groupID)->traversable;
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
      assert("check does not yet exist" && !bvhMemory.valid());
      
      context->pushActive();
      LOG("building triangles accel over "
          << children.size() << " geometries");
      
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
        d_indices  = (CUdeviceptr )tris->indexPointer;
        triangleInput.type                              = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
        triangleInput.triangleArray.vertexFormat        = OPTIX_VERTEX_FORMAT_FLOAT3;
        triangleInput.triangleArray.vertexStrideInBytes = (uint32_t)tris->vertexStride;
        triangleInput.triangleArray.numVertices         = (uint32_t)tris->vertexCount;
        triangleInput.triangleArray.vertexBuffers       = &d_vertices;
      
        triangleInput.triangleArray.indexFormat         = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
        triangleInput.triangleArray.indexStrideInBytes  = (uint32_t)tris->indexStride;
        triangleInput.triangleArray.numIndexTriplets    = (uint32_t)tris->indexCount;
        triangleInput.triangleArray.indexBuffer         = d_indices;
      
        // we always have exactly one SBT entry per shape (ie, triangle
        // mesh), and no per-primitive materials:
        triangleInput.triangleArray.flags                       = triangleInputFlags;
        triangleInput.triangleArray.numSbtRecords               = context->numRayTypes;
        triangleInput.triangleArray.sbtIndexOffsetBuffer        = 0; 
        triangleInput.triangleArray.sbtIndexOffsetSizeInBytes   = 0; 
        triangleInput.triangleArray.sbtIndexOffsetStrideInBytes = 0; 
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

      LOG_OK("successfully build triangles geom group accel");
    }
    
    void Device::sbtHitGroupsBuild(size_t maxHitProgDataSize,
                                   WriteHitProgDataCB writeHitProgDataCB,
                                   const void *callBackUserData)
    {
      LOG("building sbt hit group records");
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.hitGroupRecordsBuffer.valid())
        sbt.hitGroupRecordsBuffer.free();

      size_t numGeoms = geoms.size();
      size_t numHitGroupRecords = numGeoms * context->numRayTypes;
      size_t hitGroupRecordSize
        = OPTIX_SBT_RECORD_HEADER_SIZE
        + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxHitProgDataSize);
      assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
      sbt.hitGroupRecordSize = hitGroupRecordSize;
      sbt.hitGroupRecordCount = numHitGroupRecords;

      size_t totalHitGroupRecordsArraySize
        = numHitGroupRecords * hitGroupRecordSize;
      std::vector<uint8_t> hitGroupRecords(totalHitGroupRecordsArraySize);

      // ------------------------------------------------------------------
      // now, write all records (only on the host so far): we need to
      // write one record per geometry, per ray type
      // ------------------------------------------------------------------
      for (int geomID=0;geomID<(int)geoms.size();geomID++)
        for (int rayType=0;rayType<context->numRayTypes;rayType++) {
          // ------------------------------------------------------------------
          // compute pointer to entire record:
          // ------------------------------------------------------------------
          const int recordID = rayType + geomID*context->numRayTypes;
          uint8_t *const sbtRecord
            = hitGroupRecords.data() + recordID*hitGroupRecordSize;

          // ------------------------------------------------------------------
          // pack record header with the corresponding hit group:
          // ------------------------------------------------------------------
          // first, compute pointer to record:
          char    *const sbtRecordHeader = (char *)sbtRecord;
          // then, get gemetry we want to write (to find its hit group ID)...
          const Geom *const geom = checkGetGeom(geomID);
          // ... find the PG that goes into the record header...
          const HitGroupPG &hgPG
            = hitGroupPGs[rayType + geom->logicalHitGroupID*context->numRayTypes];
          // ... and tell optix to write that into the record
          OPTIX_CALL(SbtRecordPackHeader(hgPG.pg,sbtRecordHeader));
          
          // ------------------------------------------------------------------
          // finally, let the user fill in the record's payload using
          // the callback
          // ------------------------------------------------------------------
          uint8_t *const sbtRecordData
            = sbtRecord + OPTIX_SBT_RECORD_HEADER_SIZE;
          writeHitProgDataCB(sbtRecordData,
                                   context->owlDeviceID,
                                   geomID,
                                   rayType,
                                   callBackUserData);
        }
      sbt.hitGroupRecordsBuffer.alloc(hitGroupRecords.size());
      sbt.hitGroupRecordsBuffer.upload(hitGroupRecords);
      context->popActive();
      LOG_OK("done building (and uploading) sbt hit group records");
    }
      
    void Device::sbtRayGensBuild(size_t maxRayGenDataSize,
                                 WriteRayGenDataCB writeRayGenDataCB,
                                 const void *callBackUserData)
    {
      LOG("building sbt ray gen records");
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.rayGenRecordsBuffer.valid())
        sbt.rayGenRecordsBuffer.free();

      size_t numRayGenRecords = rayGenPGs.size();
      size_t rayGenRecordSize
        = OPTIX_SBT_RECORD_HEADER_SIZE
        + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxRayGenDataSize);
      sbt.rayGenRecordSize = rayGenRecordSize;
      sbt.rayGenRecordCount = numRayGenRecords;
      assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
      size_t totalRayGenRecordsArraySize
        = numRayGenRecords * rayGenRecordSize;
      std::vector<uint8_t> rayGenRecords(totalRayGenRecordsArraySize);

      // ------------------------------------------------------------------
      // now, write all records (only on the host so far): we need to
      // write one record per geometry, per ray type
      // ------------------------------------------------------------------
      for (int rgID=0;rgID<(int)rayGenPGs.size();rgID++) {
        // ------------------------------------------------------------------
        // compute pointer to entire record:
        // ------------------------------------------------------------------
        const int recordID = rgID;
        uint8_t *const sbtRecord
          = rayGenRecords.data() + recordID*rayGenRecordSize;
        
        // ------------------------------------------------------------------
        // pack record header with the corresponding hit group:
        // ------------------------------------------------------------------
        // first, compute pointer to record:
        char    *const sbtRecordHeader = (char *)sbtRecord;
        // ... find the PG that goes into the record header...
        const RayGenPG &rgPG
          = rayGenPGs[rgID];
        // ... and tell optix to write that into the record
        OPTIX_CALL(SbtRecordPackHeader(rgPG.pg,sbtRecordHeader));
          
        // ------------------------------------------------------------------
        // finally, let the user fill in the record's payload using
        // the callback
        // ------------------------------------------------------------------
        uint8_t *const sbtRecordData
          = sbtRecord + OPTIX_SBT_RECORD_HEADER_SIZE;
        writeRayGenDataCB(sbtRecordData,
                                context->owlDeviceID,
                                rgID,
                                callBackUserData);
      }
      sbt.rayGenRecordsBuffer.alloc(rayGenRecords.size());
      sbt.rayGenRecordsBuffer.upload(rayGenRecords);
      context->popActive();
      LOG_OK("done building (and uploading) sbt ray gen records");
    }
      
    void Device::sbtMissProgsBuild(size_t maxMissProgDataSize,
                                   WriteMissProgDataCB writeMissProgDataCB,
                                   const void *callBackUserData)
    {
      LOG("building sbt miss prog records");
      assert("check correct number of miss progs"
             && missProgPGs.size() == context->numRayTypes);
      
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.missProgRecordsBuffer.valid())
        sbt.missProgRecordsBuffer.free();

      size_t numMissProgRecords = missProgPGs.size();
      size_t missProgRecordSize
        = OPTIX_SBT_RECORD_HEADER_SIZE
        + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxMissProgDataSize);
      assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
      sbt.missProgRecordSize = missProgRecordSize;
      sbt.missProgRecordCount = numMissProgRecords;
      size_t totalMissProgRecordsArraySize
        = numMissProgRecords * missProgRecordSize;
      std::vector<uint8_t> missProgRecords(totalMissProgRecordsArraySize);

      // ------------------------------------------------------------------
      // now, write all records (only on the host so far): we need to
      // write one record per geometry, per ray type
      // ------------------------------------------------------------------
      for (int rgID=0;rgID<(int)missProgPGs.size();rgID++) {
        // ------------------------------------------------------------------
        // compute pointer to entire record:
        // ------------------------------------------------------------------
        const int recordID = rgID;
        uint8_t *const sbtRecord
          = missProgRecords.data() + recordID*missProgRecordSize;
        
        // ------------------------------------------------------------------
        // pack record header with the corresponding hit group:
        // ------------------------------------------------------------------
        // first, compute pointer to record:
        char    *const sbtRecordHeader = (char *)sbtRecord;
        // ... find the PG that goes into the record header...
        const MissProgPG &rgPG
          = missProgPGs[rgID];
        // ... and tell optix to write that into the record
        OPTIX_CALL(SbtRecordPackHeader(rgPG.pg,sbtRecordHeader));
          
        // ------------------------------------------------------------------
        // finally, let the user fill in the record's payload using
        // the callback
        // ------------------------------------------------------------------
        uint8_t *const sbtRecordData
          = sbtRecord + OPTIX_SBT_RECORD_HEADER_SIZE;
        writeMissProgDataCB(sbtRecordData,
                                  context->owlDeviceID,
                                  rgID,
                                  callBackUserData);
      }
      sbt.missProgRecordsBuffer.alloc(missProgRecords.size());
      sbt.missProgRecordsBuffer.upload(missProgRecords);
      context->popActive();
      LOG_OK("done building (and uploading) sbt miss prog records");
    }

    void Device::launch(int rgID, const vec2i &dims)
    {
      context->pushActive();
      LOG("launching ...");
      assert("check valid launch dims" && dims.x > 0);
      assert("check valid launch dims" && dims.y > 0);
      assert("check valid ray gen program ID" && rgID >= 0);
      assert("check valid ray gen program ID" && rgID <  rayGenPGs.size());

      assert("check raygen records built" && sbt.rayGenRecordCount != 0);
      sbt.sbt.raygenRecord
        = (CUdeviceptr)addPointerOffset(sbt.rayGenRecordsBuffer.get(),
                                        rgID * sbt.rayGenRecordSize);
      
      assert("check miss records built" && sbt.missProgRecordCount != 0);
      sbt.sbt.missRecordBase
        = (CUdeviceptr)sbt.missProgRecordsBuffer.get();
      sbt.sbt.missRecordStrideInBytes
        = sbt.missProgRecordSize;
      sbt.sbt.missRecordCount
        = sbt.missProgRecordCount;
      
      assert("check hit records built" && sbt.hitGroupRecordCount != 0);
      sbt.sbt.hitgroupRecordBase
        = (CUdeviceptr)sbt.hitGroupRecordsBuffer.get();
      sbt.sbt.hitgroupRecordStrideInBytes
        = sbt.hitGroupRecordSize;
      sbt.sbt.hitgroupRecordCount
        = sbt.hitGroupRecordCount;

      if (!sbt.launchParamsBuffer.valid()) {
        LOG("creating dummy launch params buffer ...");
        sbt.launchParamsBuffer.alloc(8);
      }

      // launchParamsBuffer.upload((void *)device_launchParams);
      OPTIX_CALL(Launch(context->pipeline,
                        context->stream,
                        (CUdeviceptr)sbt.launchParamsBuffer.get(),
                        sbt.launchParamsBuffer.sizeInBytes,
                        &sbt.sbt,
                        dims.x,dims.y,1
                        ));

      cudaDeviceSynchronize();
      context->popActive();
    }
    
    
  } // ::owl::ll
} //::owl
  
