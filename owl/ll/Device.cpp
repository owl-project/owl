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


      int RangeAllocator::alloc(int size)
      {
        for (int i=0;i<freedRanges.size();i++) {
          if (freedRanges[i].size >= size) {
            int where = freedRanges[i].begin;
            if (freedRanges[i].size == size)
              freedRanges.erase(freedRanges.begin()+i);
            else {
              freedRanges[i].begin += size;
              freedRanges[i].size  -= size;
            }
            return where;
          }
        }
        int where = maxAllocedID;
        maxAllocedID+=size;
        return where;
      }
      void RangeAllocator::release(int begin, int size)
      {
        for (int i=0;i<freedRanges.size();i++) {
          if (freedRanges[i].begin+freedRanges[i].size == begin) {
            begin -= freedRanges[i].size;
            size  += freedRanges[i].size;
            freedRanges.erase(freedRanges.begin()+i);
            release(begin,size);
            return;
          }
          if (begin+size == freedRanges[i].begin) {
            size  += freedRanges[i].size;
            freedRanges.erase(freedRanges.begin()+i);
            release(begin,size);
            return;
          }
        }
        if (begin+size == maxAllocedID) {
          maxAllocedID -= size;
          return;
        }
        // could not merge with any existing range: add new one
        freedRanges.push_back({begin,size});
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

      // ------------------------------------------------------------------
      // configure default module compile options
      // ------------------------------------------------------------------
      moduleCompileOptions.maxRegisterCount  = 100;
      moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
      moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;

      // ------------------------------------------------------------------
      // configure default pipeline compile options
      // ------------------------------------------------------------------
      pipelineCompileOptions = {};
      pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
      // = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
      pipelineCompileOptions.usesMotionBlur     = false;
      pipelineCompileOptions.numPayloadValues   = 2;
      pipelineCompileOptions.numAttributeValues = 2;
      pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
      pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      // pipelineCompileOptions.traversalDepth
      //   = need_tri_level
      //   ? OPTIX_TRAVERSAL_LEVEL_ANY
      //   : OPTIX_TRAVERSAL_LEVEL_TWO;
      
      // ------------------------------------------------------------------
      // configure default pipeline link options
      // ------------------------------------------------------------------
      pipelineLinkOptions.overrideUsesMotionBlur = false;
      pipelineLinkOptions.maxTraceDepth          = 2;
      pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
      
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


    void Device::geomTypeCreate(int geomTypeID,
                                size_t programDataSize)
    {
      assert(geomTypeID >= 0);
      assert(geomTypeID < geomTypes.size());
      auto &geomType = geomTypes[geomTypeID];
      assert(geomType.hitProgDataSize == (size_t)-1);
      geomType.hitProgDataSize = programDataSize;
    }

    /*! set bounding box program for given geometry type, using a
      bounding box program to be called on the device. note that
      unlike other programs (intersect, closesthit, anyhit) these
      programs are not 'per ray type', but exist only once per
      geometry type. obviously only allowed for user geometry
      typed. */
    void Device::setGeomTypeBoundsProgDevice(int geomTypeID,
                                             int moduleID,
                                             const char *progName,
                                             size_t geomDataSize)
    {
      assert(geomTypeID >= 0);
      assert(geomTypeID < geomTypes.size());
      auto &geomType = geomTypes[geomTypeID];
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      geomType.boundsProg.moduleID = moduleID;
      geomType.boundsProg.progName = progName;
      geomType.boundsProgDataSize  = geomDataSize;
    }
      
    /*! set intersect program for given geometry type and ray type
      (only allowed for user geometry types). Note progName will
      *not* be copied, so the pointer must remain valid as long as
      this geom may ever get recompiled */
    void Device::setGeomTypeIntersect(int geomTypeID,
                                      int rayTypeID,
                                      int moduleID,
                                      const char *progName)
    {
      assert(geomTypeID >= 0);
      assert(geomTypeID < geomTypes.size());
      auto &geomType = geomTypes[geomTypeID];
      
      assert(rayTypeID >= 0);
      assert(rayTypeID < context->numRayTypes);
      assert(rayTypeID < geomType.perRayType.size());
      auto &hitGroup = geomType.perRayType[rayTypeID];

      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      assert("check hitgroup isn't currently active"
             && hitGroup.pg == nullptr);
      hitGroup.intersect.moduleID = moduleID;
      hitGroup.intersect.progName = progName;
    }
    
    /*! set closest hit program for given geometry type and ray
      type. Note progName will *not* be copied, so the pointer
      must remain valid as long as this geom may ever get
      recompiled */
    void Device::setGeomTypeClosestHit(int geomTypeID,
                                       int rayTypeID,
                                       int moduleID,
                                       const char *progName)
    {
      assert(geomTypeID >= 0);
      assert(geomTypeID < geomTypes.size());
      auto &geomType = geomTypes[geomTypeID];
      
      assert(rayTypeID >= 0);
      assert(rayTypeID < context->numRayTypes);
      assert(rayTypeID < geomType.perRayType.size());
      auto &hitGroup = geomType.perRayType[rayTypeID];
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      assert("check hitgroup isn't currently active" && hitGroup.pg == nullptr);
      hitGroup.closestHit.moduleID = moduleID;
      hitGroup.closestHit.progName = progName;
    }
    
    void Device::setRayGen(int programID,
                           int moduleID,
                           const char *progName,
                           size_t dataSize)
    {
      assert(programID >= 0);
      assert(programID < rayGenPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      rayGenPGs[programID].program.moduleID = moduleID;
      rayGenPGs[programID].program.progName = progName;
      rayGenPGs[programID].program.dataSize = dataSize;
    }
    
    /*! specifies which miss program to run for a given miss prog
      ID */
    void Device::setMissProg(/*! miss program ID, in [0..numAllocatedMissProgs) */
                             int programID,
                             /*! ID of the module the program will be bound
                               in, in [0..numAllocedModules) */
                             int moduleID,
                             /*! name of the program. Note we do not NOT
                               create a copy of this string, so the string
                               has to remain valid for the duration of the
                               program */
                             const char *progName,
                             /*! size of that miss program's SBT data */
                             size_t missProgDataSize)
    {
      assert(programID >= 0);
      assert(programID < missProgPGs.size());
      
      assert(moduleID >= -1);
      assert(moduleID <  modules.size());
      assert((moduleID == -1 && progName == nullptr)
             ||
             (moduleID >= 0  && progName != nullptr));

      missProgPGs[programID].program.moduleID = moduleID;
      missProgPGs[programID].program.progName = progName;
      missProgPGs[programID].program.dataSize = missProgDataSize;
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

    std::string getNextLine(const char *&s)
    {
      std::stringstream line;
      while (*s) {
        char c = *s++;
        line << c;
        if (c == '\n') break;
      }
      return line.str();
    }
    
    inline bool ptxContainsInvalidOptixInternalCall(const std::string &line)
    {
      static const char *optix_internal_symbols[] = {
#if 1
        " _optix_",
#else
        " _optix_get_sbt",
        " _optix_trace",
        " _optix_get_world_ray_direction",
        " _optix_get_launch_index",
        " _optix_read_primitive",
        " _optix_get_payload",
#endif
        nullptr
      };
      for (const char **testSym = optix_internal_symbols; *testSym; ++testSym) {
        if (line.find(*testSym) != line.npos)
          return true;
      }
      return false;
    }
    
    std::string killAllInternalOptixSymbolsFromPtxString(const char *orignalPtxCode)
    {
      std::vector<std::string> lines;
      std::stringstream fixed;

      for (const char *s = orignalPtxCode; *s; ) {
        std::string line = getNextLine(s);
        if (ptxContainsInvalidOptixInternalCall(line))
          fixed << "//dropped: " << line;
        else
          fixed << line;
      }
      return fixed.str();
    }

    void Modules::buildOptixHandles(Context *context)
    {
      context->pushActive();
      
      assert(!modules.empty());
      LOG("building " << modules.size() << " modules");
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      
      for (int moduleID=0;moduleID<modules.size();moduleID++) {
        Module &module = modules[moduleID];
        assert(module.module == nullptr);

        // ------------------------------------------------------------------
        // first, build the *optix*-flavor of the module from that PTX string
        // ------------------------------------------------------------------

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

        // ------------------------------------------------------------------
        // now, buidl separate cuda-only module that does not contain
        // any optix-internal symbols. Note this does not actually
        // *remove* any potentially existing anyhit/closesthit/etc
        // programs in this module - it just removed all optix-related
        // calls from this module, but leaves the remaining (now
        // dysfunctional) anyhit/closesthit/etc programs still in that
        // PTX code. It woudl obviously be cleaner to completely
        // remove those programs, but that would require significantly
        // more advanced parsing of the PTX string, so right now we'll
        // just leave them in (and as it's in a module that never gets
        // used by optix, this should actually be OK)
        // ------------------------------------------------------------------
        const char *ptxCode = module.ptxCode;
        LOG("generating second, 'non-optix' version of that module, too");
        CUresult rc = (CUresult)0;
        const std::string fixedPtxCode
          = killAllInternalOptixSymbolsFromPtxString(ptxCode);
        char log[2000] = "(no log yet)";
        CUjit_option options[] = {
          CU_JIT_TARGET_FROM_CUCONTEXT,
          CU_JIT_ERROR_LOG_BUFFER,
          CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
        };
        void *optionValues[] = {
          (void*)0,
          (void*)log,
          (void*)sizeof(log)
        };
        rc = cuModuleLoadDataEx(&module.boundsModule, (void *)fixedPtxCode.c_str(),
                                3, options, optionValues);
        if (rc = CUDA_SUCCESS) {
          const char *errName = 0;
          cuGetErrorName(rc,&errName);
          PRINT(errName);
          PRINT(log);
          exit(0);
        }
        LOG_OK("created module #" << moduleID << " (both optix and cuda)");
      }
      context->popActive();
    }

    void Modules::alloc(size_t count)
    {
      assert(modules.empty());
      modules.resize(count);
    }

    void Modules::set(size_t slot,
                      const char *ptxCode)
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
      for (int geomTypeID=0;geomTypeID<geomTypes.size();geomTypeID++) {
        auto &geomType = geomTypes[geomTypeID];
        for (auto &pg : geomType.perRayType) {
          assert("check program group not already active" && pg.pg == nullptr);
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
          // ----------- intersect -----------
          Module *moduleIS = modules.get(pg.intersect.moduleID);
          std::string annotatedProgNameIS
            = pg.intersect.progName
            ? std::string("__intersection__")+pg.intersect.progName
            : "";
          if (moduleIS) {
            assert(moduleIS->module != nullptr);
            assert(pg.intersect.progName != nullptr);
            pgDesc.hitgroup.moduleIS            = moduleIS->module;
            pgDesc.hitgroup.entryFunctionNameIS = annotatedProgNameIS.c_str();
          } else {
            pgDesc.hitgroup.moduleIS            = nullptr;
            pgDesc.hitgroup.entryFunctionNameIS = nullptr;
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

        // ----------- bounds -----------
        if (geomType.boundsProg.moduleID >= 0 &&
            geomType.boundsProg.progName != nullptr) {
          LOG("building bounds function ....");
          Module *module = modules.get(geomType.boundsProg.moduleID);
          assert(module);
          assert(module->boundsModule);

          const std::string annotatedProgName
            = std::string("__boundsFuncKernel__")
            + geomType.boundsProg.progName;
          
          CUresult rc = cuModuleGetFunction(&geomType.boundsFuncKernel,
                                   module->boundsModule,
                                   annotatedProgName.c_str());
          switch(rc) {
          case CUDA_SUCCESS:
            /* all OK, nothing to do */
            LOG_OK("found bounds function " << annotatedProgName << " ... perfect!");
            break;
          case CUDA_ERROR_NOT_FOUND:
            throw std::runtime_error("in "+std::string(__PRETTY_FUNCTION__)
                                     +": could not find OPTIX_BOUNDS_PROGRAM("
                                     +geomType.boundsProg.progName+")");
          default:
            const char *errName = 0;
            cuGetErrorName(rc,&errName);
            PRINT(errName);
            exit(0);
          }
        }
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
      for (auto &geomType : geomTypes) 
        for (auto &pg : geomType.perRayType) {
          if (pg.pg) optixProgramGroupDestroy(pg.pg);
          pg.pg = nullptr;
        }
      // ---------------------- miss ----------------------
      for (auto &pg : missProgPGs) {
        if (pg.pg) optixProgramGroupDestroy(pg.pg);
        pg.pg = nullptr;
      }
    }

    void Device::allocGeomTypes(size_t count)
    {
      assert(geomTypes.empty());
      geomTypes.resize(count);
      for (auto &gt : geomTypes) {
        if (gt.perRayType.empty())
          gt.perRayType.resize(context->numRayTypes);
        assert(gt.perRayType.size() == context->numRayTypes);
      }
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
      assert(!device->geomTypes.empty());
      for (auto &geomType : device->geomTypes)
        for (auto &pg : geomType.perRayType)
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
    
    /*! set a buffer of bounding boxes that this user geometry will
      use when building the accel structure. this is one of
      multiple ways of specifying the bounding boxes for a user
      gometry (the other two being a) setting the geometry type's
      boundsFunc, or b) setting a host-callback fr computing the
      bounds). Only one of the three methods can be set at any
      given time */
    void Device::userGeomSetBoundsBuffer(int geomID,
                                         int bufferID)
    {
      UserGeom *user
        = checkGetUserGeom(geomID);
      assert("double-check valid geom" && user);
      
      Buffer   *buffer
        = checkGetBuffer(bufferID);
      assert("double-check valid buffer" && buffer);
      size_t offset = 0; // don't support offset/stride yet
      user->d_boundsMemory = addPointerOffset(buffer->get(),offset);
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
      
        // we always have exactly one SBT entry per shape (ie, triangle
        // mesh), and no per-primitive materials:
        ta.flags                       = triangleInputFlags;
        ta.numSbtRecords               = context->numRayTypes;
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
    








    void Device::sbtHitProgsBuild(WriteHitProgDataCB writeHitProgDataCB,
                                  const void *callBackUserData)
    {
      LOG("building sbt hit group records");
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.hitGroupRecordsBuffer.valid())
        sbt.hitGroupRecordsBuffer.free();

      size_t maxHitProgDataSize = 0;
      for (int geomID=0;geomID<geoms.size();geomID++) {
        Geom *geom = geoms[geomID];
        if (!geom) continue;
        GeomType &gt = geomTypes[geom->geomTypeID];
        maxHitProgDataSize = std::max(maxHitProgDataSize,gt.hitProgDataSize);
      }
      PRINT(maxHitProgDataSize);
      assert("make sure all geoms had their program size set"
             && maxHitProgDataSize != (size_t)-1);
      size_t numHitGroupEntries = sbt.rangeAllocator.maxAllocedID;
      size_t numHitGroupRecords = numHitGroupEntries*context->numRayTypes;
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
      PING;
      PRINT(groups.size());
      for (auto group : groups) {
        if (!group) continue;
        if (!group->containsGeom()) continue;
        GeomGroup *gg = (GeomGroup *)group;
        const int sbtOffset = gg->sbtOffset;
        PRINT(sbtOffset);
        PRINT(gg->children.size());
        for (int childID=0;childID<gg->children.size();childID++) {
          Geom *geom = gg->children[childID];
          if (!geom) continue;
          
          const int geomID    = geom->geomID;
          for (int rayTypeID=0;rayTypeID<context->numRayTypes;rayTypeID++) {
            // ------------------------------------------------------------------
            // compute pointer to entire record:
            // ------------------------------------------------------------------
            const int recordID
              = (sbtOffset+childID)*context->numRayTypes + rayTypeID;
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
            auto &geomType = geomTypes[geom->geomTypeID];
            const HitGroupPG &hgPG
              = geomType.perRayType[rayTypeID];
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
                               rayTypeID,
                               callBackUserData);
          }
        }
      }
      PRINT(hitGroupRecords.size());
      sbt.hitGroupRecordsBuffer.alloc(hitGroupRecords.size());
      sbt.hitGroupRecordsBuffer.upload(hitGroupRecords);
      context->popActive();
      PING;
      LOG_OK("done building (and uploading) sbt hit group records");
    }
      
    void Device::sbtRayGensBuild(WriteRayGenDataCB writeRayGenDataCB,
                                 const void *callBackUserData)
    {
      LOG("building sbt ray gen records");
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.rayGenRecordsBuffer.valid())
        sbt.rayGenRecordsBuffer.free();

      size_t maxRayGenDataSize = 0;
      for (int rgID=0;rgID<(int)rayGenPGs.size();rgID++) 
        maxRayGenDataSize = std::max(maxRayGenDataSize,
                                     rayGenPGs[rgID].program.dataSize);
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
      
    void Device::sbtMissProgsBuild(WriteMissProgDataCB writeMissProgDataCB,
                                   const void *callBackUserData)
    {
      LOG("building sbt miss prog records");
      assert("check correct number of miss progs"
             && missProgPGs.size() == context->numRayTypes);
      
      context->pushActive();
      // TODO: move this to explicit destroyhitgroups
      if (sbt.missProgRecordsBuffer.valid())
        sbt.missProgRecordsBuffer.free();

      size_t maxMissProgDataSize = 0;
      for (int mpID=0;mpID<(int)missProgPGs.size();mpID++) 
        maxMissProgDataSize = std::max(maxMissProgDataSize,
                                     rayGenPGs[mpID].program.dataSize);
      
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
        = (uint32_t)sbt.missProgRecordSize;
      sbt.sbt.missRecordCount
        = (uint32_t)sbt.missProgRecordCount;
      
      assert("check hit records built" && sbt.hitGroupRecordCount != 0);
      sbt.sbt.hitgroupRecordBase
        = (CUdeviceptr)sbt.hitGroupRecordsBuffer.get();
      sbt.sbt.hitgroupRecordStrideInBytes
        = (uint32_t)sbt.hitGroupRecordSize;
      sbt.sbt.hitgroupRecordCount
        = (uint32_t)sbt.hitGroupRecordCount;

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
    
    void Device::createUserGeom(int geomID,
                                /*! the "logical" hit group ID:
                                  will always count 0,1,2... evne
                                  if we are using multiple ray
                                  types; the actual hit group
                                  used when building the SBT will
                                  then be 'geomTypeID *
                                  numRayTypes) */
                                int geomTypeID,
                                int numPrims)
    {
      assert("check ID is valid" && geomID >= 0);
      assert("check ID is valid" && geomID < geoms.size());
      assert("check given ID isn't still in use" && geoms[geomID] == nullptr);

      assert("check valid hit group ID" && geomTypeID >= 0);
      assert("check valid hit group ID" && geomTypeID <  geomTypes.size());
        
      geoms[geomID] = new UserGeom(geomID,geomTypeID,numPrims);
      assert("check 'new' was successful" && geoms[geomID] != nullptr);
    }
    
    void Device::createTrianglesGeom(int geomID,
                                     /*! the "logical" hit group ID:
                                       will always count 0,1,2... evne
                                       if we are using multiple ray
                                       types; the actual hit group
                                       used when building the SBT will
                                       then be 'geomTypeID *
                                       numRayTypes) */
                                     int geomTypeID)
    {
      assert("check ID is valid" && geomID >= 0);
      assert("check ID is valid" && geomID < geoms.size());
      assert("check given ID isn't still in use" && geoms[geomID] == nullptr);

      assert("check valid hit group ID" && geomTypeID >= 0);
      assert("check valid hit group ID" && geomTypeID < geomTypes.size());
        
      geoms[geomID] = new TrianglesGeom(geomID,geomTypeID);
      assert("check 'new' was successful" && geoms[geomID] != nullptr);
    }

    /*! resize the array of geom IDs. this can be either a
      'grow' or a 'shrink', but 'shrink' is only allowed if all
      geoms that would get 'lost' have alreay been
      destroyed */
    void Device::reallocGroups(size_t newCount)
    {
      for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)groups.size();idxWeWouldLose++)
        assert("realloc would lose a geom that was not properly destroyed" &&
               groups[idxWeWouldLose] == nullptr);
      groups.resize(newCount);
    }
    /*! resize the array of buffer handles. this can be either a
      'grow' or a 'shrink', but 'shrink' is only allowed if all
      buffer handles that would get 'lost' have alreay been
      destroyed */
    void Device::reallocBuffers(size_t newCount)
    {
      for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)buffers.size();idxWeWouldLose++)
        assert("realloc would lose a geom that was not properly destroyed" &&
               buffers[idxWeWouldLose] == nullptr);
      buffers.resize(newCount);
    }

    void Device::createTrianglesGeomGroup(int groupID,
                                          int *geomIDs,
                                          int geomCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] ==nullptr);
        
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && geomCount == 0) ||
              (geomIDs != nullptr && geomCount >  0)));
        
      TrianglesGeomGroup *group
        = new TrianglesGeomGroup(geomCount,
                                 sbt.rangeAllocator.alloc(geomCount));
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      // set children - todo: move to separate (api?) function(s)!?
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
    
    void Device::createUserGeomGroup(int groupID,
                                     int *geomIDs,
                                     int geomCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] ==nullptr);
        
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && geomCount == 0) ||
              (geomIDs != nullptr && geomCount >  0)));
        
      UserGeomGroup *group
        = new UserGeomGroup(geomCount,
                            sbt.rangeAllocator.alloc(geomCount));
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      // set children - todo: move to separate (api?) function(s)!?
      for (int childID=0;childID<geomCount;childID++) {
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



    void Device::groupBuildPrimitiveBounds(int groupID,
                                           size_t maxGeomDataSize,
                                           WriteUserGeomBoundsDataCB cb,
                                           void *cbData)
    {
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
    }
    
  } // ::owl::ll
} //::owl
  
