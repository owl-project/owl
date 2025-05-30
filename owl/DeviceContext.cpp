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

#include "Context.h"
#include "UserGeom.h"
#include "InstanceGroup.h"
#include "CUDADriver.h"

#include <optix_function_table_definition.h>

#define LOG(message)                            \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (Context::logging())                       \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl: "                               \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {

  /*! logging callback passed to optix for intercepting optix log messages */
  static void context_log_cb(unsigned int level,
                             const char *tag,
                             const char *message,
                             void *)
  {
    if (level == 1 || level == 2)
      fprintf( stderr, "[%2d][%12s]: %s\n", (int)level, tag, message );
  }

  /*! allocate 'size' consecutive SBT entries, and return index of
      first of those */
  int RangeAllocator::alloc(size_t size)
  {
    for (size_t i=0;i<freedRanges.size();i++) {
      if (freedRanges[i].size >= size) {
        size_t where = freedRanges[i].begin;
        if (freedRanges[i].size == size)
          freedRanges.erase(freedRanges.begin()+i);
        else {
          freedRanges[i].begin += size;
            freedRanges[i].size  -= size;
        }
        return (int)where;
      }
    }
    size_t where = maxAllocedID;
    maxAllocedID+=size;
    assert(maxAllocedID == size_t(int(maxAllocedID)));
    return (int)where;
  }

  /*! a given group has died, and tells us to release given range
      (starting at begin, with 'siez' elements', to be re-used when
      appropriate */
  void RangeAllocator::release(size_t begin, size_t size)
  {
    for (size_t i=0;i<freedRanges.size();i++) {
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

  
  /*! creates the N device contexts with the given device IDs. If list
    of device is nullptr, and number requested devices is > 1, then
    the first N devices will get used; invalid device IDs in the
    list will automatically get dropped */
  std::vector<DeviceContext::SP> createDeviceContexts(Context *parent,
                                                      int32_t *deviceIDs,
                                                      int      numDevices)
  {
    LOG("context ramping up - creating low-level devicegroup");

    // ------------------------------------------------------------------
    // init cuda, and error-out if no cuda devices exist
    // ------------------------------------------------------------------
    LOG("initializing CUDA");
    cudaFree(0);
    
    int totalNumDevicesAvailable = 0;
    OWL_CUDA_CALL(GetDeviceCount(&totalNumDevicesAvailable));
    if (totalNumDevicesAvailable == 0)
      OWL_RAISE("#owl: no CUDA capable devices found!");
    LOG_OK("found " << totalNumDevicesAvailable << " CUDA device(s)");


    // ------------------------------------------------------------------
    // fill in deviceIDs if not provided
    // ------------------------------------------------------------------
    std::vector<int> tmpDeviceIDs;
    if (deviceIDs == nullptr) {
      if (numDevices <= 0)
        numDevices = totalNumDevicesAvailable;
      
      for (int i=0;i<numDevices;i++)
        tmpDeviceIDs.push_back(i);
      deviceIDs = tmpDeviceIDs.data();
    } else {
      assert(numDevices > 0);
      for (int i=0;i<numDevices;i++)
        assert(deviceIDs[i] >= 0 && deviceIDs[i] < totalNumDevicesAvailable);
    }
    
    
    // ------------------------------------------------------------------
    // init optix itself
    // ------------------------------------------------------------------

    int major =  OPTIX_VERSION/10000;
    int minor = (OPTIX_VERSION%10000)/100;
    int micro =  OPTIX_VERSION%100;
    
    std::stringstream ss;
    ss << "initializing optix " << major << "." << minor << "." << micro;
    LOG(ss.str().c_str());
    static bool initialized = false;
    if (!initialized) {
      OPTIX_CHECK(optixInit());
      initialized = true;
    }
    
    // from here on, we need a non-empty list of requested device IDs
    assert(deviceIDs != nullptr && numDevices > 0);
    
    // ------------------------------------------------------------------
    // create actual devices, ignoring those that failed to initialize
    // ------------------------------------------------------------------
    std::vector<DeviceContext::SP> devices;
    for (int i=0;i<numDevices;i++) {
      try {
        DeviceContext::SP dev = std::make_shared<DeviceContext>(parent,i,deviceIDs[i]);
        assert(dev);
        devices.push_back(dev);
      } catch (std::exception &e) {
        std::cout << OWL_TERMINAL_RED
                  << "#owl: Error creating optix device on CUDA device #"
                  << deviceIDs[i] << ": " << e.what() << " ... dropping this device"
                  << OWL_TERMINAL_DEFAULT << std::endl;
      }
    }
    
    // ------------------------------------------------------------------
    // some final sanity check that we managed to create at least
    // one device...
    // ------------------------------------------------------------------
    if (devices.empty())
      OWL_RAISE("fatal error - could not find/create any optix devices");
    
    LOG_OK("successfully created device group with " << devices.size() << " devices");
    return devices;
  }
  
  
  
  DeviceContext::DeviceContext(Context *parent,
                               int owlID,
                               int cudaID)
    : parent(parent),
      ID(owlID),
      cudaDeviceID(cudaID)
  {
    LOG("trying to create owl device on CUDA device #" << cudaDeviceID);
    
    LOG(" - device: " << getDeviceName());
    
    OWL_CUDA_CHECK(cudaSetDevice(cudaDeviceID));
    OWL_CUDA_CHECK(cudaStreamCreate(&stream));

#if 1
    // use current context
    cudaContext = 0;
#else
    CUresult  cuRes = _cuCtxGetCurrent(&cudaContext);
    if (cuRes != CUDA_SUCCESS) 
       OWL_RAISE("Error querying current CUDA context...");
#endif
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,context_log_cb,this,4));
  }

  DeviceContext::~DeviceContext()
  {
    destroyMissPrograms();
    destroyRayGenPrograms();
    destroyHitGroupPrograms();
    destroyPrograms();
    destroyPipeline();
    
    OPTIX_CHECK(optixDeviceContextDestroy(optixContext));
    cudaStreamDestroy(stream);
  }
  
  
  /*! return CUDA's name string for given device */
  std::string DeviceContext::getDeviceName() const
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, getCudaDeviceID());
    return prop.name;
  }
  
  /*! helper function - return cuda device ID of this device */
  int DeviceContext::getCudaDeviceID() const
  {
    return cudaDeviceID;
  }
  
  void DeviceContext::destroyPipeline()
  {
    if (!pipeline) return;
    
    SetActiveGPU forLifeTime(this);
    
    OPTIX_CHECK(optixPipelineDestroy(pipeline));
    pipeline = 0;
  }
  
  
  void DeviceContext::configurePipelineOptions(bool debug)
  {
    // ------------------------------------------------------------------
    // configure default module compile options
    // ------------------------------------------------------------------
  if (!debug) {
    moduleCompileOptions.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_3;
#if OPTIX_VERSION >= 70400
    // 7.4 no longer has 'lineinfo'
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
  } 
  else {
    std::cout << "WARNING: RUNNING OPTIX PROGRAMS IN -O0 DEBUG MODE!!!" << std::endl;
    moduleCompileOptions.maxRegisterCount  = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT ;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
#if OPTIX_VERSION >= 70400
    // 7.4 no longer has 'lineinfo'
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;
#else
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;
#endif
  }

#if OPTIX_VERSION >= 70200
    // Bound values of launch params
    moduleCompileOptions.boundValues = parent->boundLaunchParamValues.data();
    moduleCompileOptions.numBoundValues = (unsigned int)parent->boundLaunchParamValues.size();
#endif
    
    // ------------------------------------------------------------------
    // configure default pipeline compile options
    // ------------------------------------------------------------------
    pipelineCompileOptions = {};
    assert(parent->maxInstancingDepth >= 0);
    switch (parent->maxInstancingDepth) {
    case 0:
      pipelineCompileOptions.traversableGraphFlags
        = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
      break;
    case 1:
      pipelineCompileOptions.traversableGraphFlags
        = parent->motionBlurEnabled
        ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY
        : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING
        ;
      break;
    default:
      pipelineCompileOptions.traversableGraphFlags
        = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_ANY;
      break;
    }
    pipelineCompileOptions.usesMotionBlur     = parent->motionBlurEnabled;
    pipelineCompileOptions.numPayloadValues   = parent->numPayloadValues;
    pipelineCompileOptions.numAttributeValues = parent->numAttributeValues;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_TRIANGLE;
    pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_CUSTOM;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
    
    // ------------------------------------------------------------------
    // configure default pipeline link options
    // ------------------------------------------------------------------
    // pipelineLinkOptions.overrideUsesMotionBlur = motionBlurEnabled;
    pipelineLinkOptions.maxTraceDepth          = 2;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";

    // if (parent->curvesEnabled) 
#if OWL_CAN_DO_CURVES
   pipelineCompileOptions.usesPrimitiveTypeFlags
        |= OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR
        | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE
        | OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
#endif

#if OWL_CAN_DO_SPHERES
   pipelineCompileOptions.usesPrimitiveTypeFlags |= OPTIX_PRIMITIVE_TYPE_FLAGS_SPHERE;
#endif

// #if 1
//     std::cout << "HACK: FORCE-ENABLE CURVES!" << std::endl;
//     int degree = 1;
//     bool motion_blur = 0;
//     switch( degree )
//       {
//       case 1:
//         pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_LINEAR;
//         break;
//       case 2:
//         pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_QUADRATIC_BSPLINE;
//         break;
//       case 3:
//         pipelineCompileOptions.usesPrimitiveTypeFlags = OPTIX_PRIMITIVE_TYPE_FLAGS_ROUND_CUBIC_BSPLINE;
//         break;
//       }
// #endif
  }
  
  void DeviceContext::buildPipeline()
  {
    SetActiveGPU forLifeTime(this);
    
    auto &allPGs = allActivePrograms;
    if (allPGs.empty())
      OWL_RAISE("trying to create a pipeline w/ 0 programs!?");
    
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
    
    uint32_t maxAllowedByOptix = 0;
    optixDeviceContextGetProperty
      (optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
       &maxAllowedByOptix,
       sizeof(maxAllowedByOptix));
    if (uint32_t(parent->maxInstancingDepth+1) > maxAllowedByOptix)
      OWL_RAISE
        ("error when building pipeline: "
         "attempting to set max instancing depth to "
         "value that exceeds OptiX's MAX_TRAVERSABLE_GRAPH_DEPTH limit");
    
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
                 int(parent->maxInstancingDepth+1)
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 ));
  }

  void DeviceContext::buildPrograms()
  {
    SetActiveGPU forLifeTime(this);
    destroyPrograms();
    buildMissPrograms();
    buildRayGenPrograms();
    buildHitGroupPrograms();
  }

  void DeviceContext::buildCurvesModules()
  {
#if OWL_CAN_DO_CURVES 
    SetActiveGPU forLifeTime(this);

    for (int forceCap=0;forceCap<2;forceCap++) {
      for (int degree=1;degree<=3;degree++) {
        if (curvesModule[forceCap][degree-1] != nullptr)
          optixModuleDestroy(curvesModule[forceCap][degree-1]);
      
        OptixBuiltinISOptions builtinISOptions = {};
        switch (degree) {
        case 1:
          builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR;
          break;
        case 2:
          builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE;
          break;
        case 3:
          builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE;
          break;
        }
        builtinISOptions.usesMotionBlur = parent->motionBlurEnabled;  // enable motion-blur for built-in intersector
        builtinISOptions.curveEndcapFlags = forceCap;
      
        OPTIX_CHECK(optixBuiltinISModuleGet(optixContext, &moduleCompileOptions, &pipelineCompileOptions,
                                            &builtinISOptions, &curvesModule[forceCap][degree-1]));
      }
    }
#endif
  }

  void DeviceContext::buildSphereModule()
  {
#if OWL_CAN_DO_SPHERES
    SetActiveGPU forLifeTime(this);

    if (spheresModule != nullptr)
        optixModuleDestroy(spheresModule);
      
    OptixBuiltinISOptions builtinISOptions = {};
    builtinISOptions.builtinISModuleType = OPTIX_PRIMITIVE_TYPE_SPHERE;
    builtinISOptions.usesMotionBlur = parent->motionBlurEnabled;  // enable motion-blur for built-in intersector
      
    OPTIX_CHECK(optixBuiltinISModuleGet(optixContext, &moduleCompileOptions, &pipelineCompileOptions,
                                            &builtinISOptions, &spheresModule));
#endif
  }
  
  void DeviceContext::destroyPrograms()
  {
    SetActiveGPU forLifeTime(this);
    destroyPipeline();
    destroyMissPrograms();
    destroyRayGenPrograms();
    destroyHitGroupPrograms();

    allActivePrograms.clear();
  }

  /*! build all optix progrmas for miss program types */
  void DeviceContext::buildMissPrograms()
  {
    for (size_t progID=0;progID<parent->missProgTypes.size();progID++) {
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc    pgDesc    = {};
      
      MissProgType *prog = parent->missProgTypes.getPtr(progID);
      if (!prog) continue;
      auto &dd = prog->getDD(shared_from_this());
      assert(dd.pg == 0);

      Module::SP module = prog->module;
      assert(module);
      
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      OptixModule optixModule = module->getDD(shared_from_this()).module;
      assert(optixModule);
      
      pgDesc.miss.module            = optixModule;
      pgDesc.miss.entryFunctionName = prog->annotatedProgName.c_str();
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                          &pgDesc,
                                          1,
                                          &pgOptions,
                                          log,&sizeof_log,
                                          &dd.pg
                                          ));
      assert(dd.pg);
      allActivePrograms.push_back(dd.pg);
    }
  }

  void DeviceContext::destroyMissPrograms()
  {
    for (size_t progID=0;progID<parent->missProgTypes.size();progID++) {
      MissProgType *prog = parent->missProgTypes.getPtr(progID);
      if (!prog) continue;
      auto &dd = prog->getDD(shared_from_this());
      if (dd.pg == 0) continue;

      OPTIX_CHECK(optixProgramGroupDestroy(dd.pg));
      dd.pg = 0;
    }
  }
  
  void DeviceContext::buildRayGenPrograms()
  {
    for (size_t pgID=0;pgID<parent->rayGenTypes.size();pgID++) {
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc    pgDesc    = {};
      
      RayGenType *prog = parent->rayGenTypes.getPtr(pgID);
      if (!prog) continue;
      
      auto &dd = prog->getDD(shared_from_this());
      assert(dd.pg == 0);
      
      Module::SP module = prog->module;
      assert(module);
      
      OptixModule optixModule = module->getDD(shared_from_this()).module;
      assert(optixModule);
      
      pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      pgDesc.raygen.module            = optixModule;
      pgDesc.raygen.entryFunctionName = prog->annotatedProgName.c_str();
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                          &pgDesc,
                                          1,
                                          &pgOptions,
                                          log,&sizeof_log,
                                          &dd.pg
                                          ));
      assert(dd.pg);
      allActivePrograms.push_back(dd.pg);
    }
  }
  
  void DeviceContext::destroyRayGenPrograms()
  {
    for (size_t pgID=0;pgID<parent->rayGenTypes.size();pgID++) {
      RayGenType *prog = parent->rayGenTypes.getPtr(pgID);
      if (!prog) continue;
      
      auto &dd = prog->getDD(shared_from_this());
      if (dd.pg == 0) continue;
      
      OPTIX_CHECK(optixProgramGroupDestroy(dd.pg));
      dd.pg = 0;
    }
  }
  
  void DeviceContext::buildHitGroupPrograms()
  {
    assert(parent);
    const int numRayTypes = parent->numRayTypes;
    
    // ------------------------------------------------------------------
    // geometry type programs -> what goes into hit groups
    // ------------------------------------------------------------------
    for (size_t geomTypeID=0;geomTypeID<parent->geomTypes.size();geomTypeID++) {
      GeomType::SP geomType = parent->geomTypes.getSP(geomTypeID);
      if (!geomType)
        continue;
      
      UserGeomType::SP userGeomType
        = geomType->as<UserGeomType>();
      if (userGeomType && parent->motionBlurEnabled)
        userGeomType->buildMotionBoundsProg();
      else if (userGeomType)
        userGeomType->buildBoundsProg();

      auto &dd = geomType->getDD(shared_from_this());
      dd.hgPGs.clear();
      dd.hgPGs.resize(numRayTypes);

      for (int rt=0;rt<numRayTypes;rt++) {
        
        OptixProgramGroupOptions pgOptions = {};
        OptixProgramGroupDesc    pgDesc    = {};
        
        pgDesc.kind      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        // ----------- default init closesthit -----------
        pgDesc.hitgroup.moduleCH            = nullptr;
        pgDesc.hitgroup.entryFunctionNameCH = nullptr;
        // ----------- default init anyhit -----------
        pgDesc.hitgroup.moduleAH            = nullptr;
        pgDesc.hitgroup.entryFunctionNameAH = nullptr;
        // ----------- default init intersect -----------
        pgDesc.hitgroup.moduleIS            = nullptr;
        pgDesc.hitgroup.entryFunctionNameIS = nullptr;
        
        // now let the type fill in what it has
        dd.fillPGDesc(pgDesc,geomType.get(),rt);

        char log[2048];
        size_t sizeof_log = sizeof( log );
        OptixProgramGroup &pg = dd.hgPGs[rt];
        OPTIX_CHECK(optixProgramGroupCreate(optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &pg
                                            ));
        allActivePrograms.push_back(pg);
      }
    }

    // ------------------------------------------------------------------
    // instance programs -> what goes into instances
    // ------------------------------------------------------------------
    for (size_t groupID=0;groupID<parent->groups.size();groupID++) {
      // skip groups which are not "Instance Group" types
      Group::SP group = parent->groups.getSP(groupID);
      if (!group)
        continue;
      InstanceGroup::SP instanceGroup
        = group->as<InstanceGroup>();
      
      if (!instanceGroup)
        continue;
      
      if (parent->motionBlurEnabled)
        instanceGroup->buildMotionInstanceProg();
      else
        instanceGroup->buildInstanceProg();
    }
  }
  
  void DeviceContext::destroyHitGroupPrograms()
  {
    for (size_t geomTypeID=0;geomTypeID<parent->geomTypes.size();geomTypeID++) {
      GeomType::SP geomType = parent->geomTypes.getSP(geomTypeID);
      if (!geomType)
        continue;

      auto &dd = geomType->getDD(shared_from_this());
      for (auto &pg : dd.hgPGs) 
        if (pg) {
          OPTIX_CHECK(optixProgramGroupDestroy(pg));
        }
      dd.hgPGs.clear();
    }
  }
  
} // ::owl
