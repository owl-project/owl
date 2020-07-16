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

#include "Context.h"
#include "Module.h"
#include "Geometry.h"
#include "api/Triangles.h"
#include "api/UserGeom.h"
#include "Texture.h"
#include "api/TrianglesGeomGroup.h"
#include "api/UserGeomGroup.h"

#include "owl/ll/Device.h"

#define LOG(message)                            \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {

  Context::~Context()
  {
  }
  
  Context::SP Context::create(int32_t *requestedDeviceIDs,
                              int      numRequestedDevices)
  {
    LOG("creating node-graph context");
    return std::make_shared<Context>(requestedDeviceIDs,
                                     numRequestedDevices);
  }
  
  Context::Context(int32_t *requestedDeviceIDs,
                   int      numRequestedDevices)
    : buffers(this),
      textures(this),
      groups(this),
      rayGenTypes(this),
      rayGens(this),
      launchParamTypes(this),
      launchParams(this),
      missProgTypes(this),
      missProgs(this),
      geomTypes(this),
      geoms(this),
      modules(this)
  {
    LOG("context ramping up - creating low-level devicegroup");
    // ll = ll::DeviceGroup::create();
    llo = ll::DeviceGroup::create(requestedDeviceIDs,
                                  numRequestedDevices);
    createDeviceData(getDevices());
    LOG_OK("device group created");


    LaunchParamsType::SP emptyLPType
      = createLaunchParamsType(0,{});
    dummyLaunchParams = createLaunchParams(emptyLPType);
  }
  
  /*! creates a buffer that uses CUDA host pinned memory; that
    memory is pinned on the host and accessive to all devices in the
    device group */
  Buffer::SP Context::hostPinnedBufferCreate(OWLDataType type,
                                             size_t count)
  {
    Buffer::SP buffer = std::make_shared<HostPinnedBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(llo->devices);
    buffer->resize(count);
    return buffer;
  }

  /*! creates a buffer that uses CUDA managed memory; that memory is
    managed by CUDA (see CUDAs documentatoin on managed memory) and
    accessive to all devices in the deviec group */
  Buffer::SP
  Context::managedMemoryBufferCreate(OWLDataType type,
                                     size_t count,
                                     const void *init)
  {
    Buffer::SP buffer
      = std::make_shared<ManagedMemoryBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(llo->devices);
    buffer->resize(count);
    if (init)
      buffer->upload(init);
    return buffer;
  }
  
  Buffer::SP Context::deviceBufferCreate(OWLDataType type,
                                         size_t count,
                                         const void *init)
  {
    Buffer::SP buffer
      = std::make_shared<DeviceBuffer>(this,type);
    assert(buffer);
    buffer->createDeviceData(llo->devices);
    buffer->resize(count);
    if (init)
      buffer->upload(init);
    return buffer;
  }

  Texture::SP
  Context::texture2DCreate(OWLTexelFormat texelFormat,
                           OWLTextureFilterMode filterMode,
                           const vec2i size,
                           uint32_t linePitchInBytes,
                           const void *texels)
  {
    Texture::SP texture
      = std::make_shared<Texture>(this,size,linePitchInBytes,texelFormat,filterMode,texels);
    assert(texture);
    return texture;
  }
    

  Buffer::SP
  Context::graphicsBufferCreate(OWLDataType type,
                                size_t count,
                                cudaGraphicsResource_t resource)
  {
    Buffer::SP buffer
      = std::make_shared<GraphicsBuffer>(this, type, resource);
    
    assert(buffer);
    buffer->createDeviceData(getDevices());
    buffer->resize(count);

    return buffer;
  }

  RayGen::SP
  Context::createRayGen(const std::shared_ptr<RayGenType> &type)
  {
    RayGen::SP rg = std::make_shared<RayGen>(this,type);
    rg->createDeviceData(getDevices());
    return rg;
  }

  LaunchParams::SP
  Context::createLaunchParams(const std::shared_ptr<LaunchParamsType> &type)
  {
    LaunchParams::SP lp = std::make_shared<LaunchParams>(this,type);
    lp->createDeviceData(getDevices());
    return lp;
  }

  MissProg::SP
  Context::createMissProg(const std::shared_ptr<MissProgType> &type)
  {
    MissProg::SP mp = std::make_shared<MissProg>(this,type);
    mp->createDeviceData(getDevices());
    return mp;
  }

  RayGenType::SP
  Context::createRayGenType(Module::SP module,
                            const std::string &progName,
                            size_t varStructSize,
                            const std::vector<OWLVarDecl> &varDecls)
  {
    RayGenType::SP rgt = std::make_shared<RayGenType>(this,
                                                      module,progName,
                                                      varStructSize,
                                                      varDecls);
    rgt->createDeviceData(getDevices());
    return rgt;
  }
  

  LaunchParamsType::SP
  Context::createLaunchParamsType(size_t varStructSize,
                                  const std::vector<OWLVarDecl> &varDecls)
  {
    LaunchParamsType::SP lpt
      = std::make_shared<LaunchParamsType>(this,
                                              varStructSize,
                                           varDecls);
    lpt->createDeviceData(getDevices());
    return lpt;
  }
  

  MissProgType::SP
  Context::createMissProgType(Module::SP module,
                              const std::string &progName,
                              size_t varStructSize,
                              const std::vector<OWLVarDecl> &varDecls)
  {
    MissProgType::SP mpt
      = std::make_shared<MissProgType>(this,
                                       module,progName,
                                       varStructSize,
                                       varDecls);
    mpt->createDeviceData(getDevices());
    return mpt;
  }
  

  GeomGroup::SP Context::trianglesGeomGroupCreate(size_t numChildren)
  {
    GeomGroup::SP gg
      = std::make_shared<TrianglesGeomGroup>(this,numChildren);
    gg->createDeviceData(getDevices());
    return gg;
  }

  GeomGroup::SP Context::userGeomGroupCreate(size_t numChildren)
  {
    GeomGroup::SP gg
      = std::make_shared<UserGeomGroup>(this,numChildren);
    gg->createDeviceData(getDevices());
    return gg;
  }


  GeomType::SP
  Context::createGeomType(OWLGeomKind kind,
                          size_t varStructSize,
                          const std::vector<OWLVarDecl> &varDecls)
  {
    GeomType::SP gt;
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      gt = std::make_shared<TrianglesGeomType>(this,varStructSize,varDecls);
      break;
    case OWL_GEOMETRY_USER:
      gt = std::make_shared<UserGeomType>(this,varStructSize,varDecls);
      break;
    default:
      OWL_NOTIMPLEMENTED;
    }
    gt->createDeviceData(getDevices());
    return gt;
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    Module::SP module = std::make_shared<Module>(this,ptxCode);//,modules.allocID());;
    assert(module);
    module->createDeviceData(getDevices());
    return module;
  }

  std::shared_ptr<Geom> UserGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<UserGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  std::shared_ptr<Geom> TrianglesGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<TrianglesGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  void Context::buildHitGroupRecordsOn(ll::Device *device)
  {
    LOG("building SBT hit group records");
    int oldActive = device->pushActive();
    // TODO: move this to explicit destroyhitgroups
    if (device->sbt.hitGroupRecordsBuffer.alloced())
      device->sbt.hitGroupRecordsBuffer.free();

    size_t maxHitProgDataSize = 0;
    for (int i=0;i<geoms.size();i++) {
      Geom *geom = (Geom *)geoms.getPtr(i);
      if (!geom) continue;
        
      assert(geom->geomType);
      maxHitProgDataSize = std::max(maxHitProgDataSize,geom->geomType->varStructSize);
    }
    // for (int geomID=0;geomID<geoms.size();geomID++) {
    //   Geom *geom = geoms[geomID];
    //   if (!geom) continue;
    //   GeomType &gt = geomTypes[geom->geomTypeID];
    //   maxHitProgDataSize = std::max(maxHitProgDataSize,gt.hitProgDataSize);
    // }
    PING; PRINT(maxHitProgDataSize);
      
    // if (maxHitProgDataSize == size_t(-1))
    //   throw std::runtime_error("in sbtHitProgsBuild: at least on geometry uses a type for which geomTypeCreate has not been called");
    // assert("make sure all geoms had their program size set"
    //        && maxHitProgDataSize != (size_t)-1);
    size_t numHitGroupEntries = sbtRangeAllocator.maxAllocedID;
    PRINT(numHitGroupEntries);
    size_t numHitGroupRecords = numHitGroupEntries*numRayTypes + 1;
    size_t hitGroupRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxHitProgDataSize);
    PRINT(hitGroupRecordSize);
    
    assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    device->sbt.hitGroupRecordSize = hitGroupRecordSize;
    device->sbt.hitGroupRecordCount = numHitGroupRecords;

    size_t totalHitGroupRecordsArraySize
      = numHitGroupRecords * hitGroupRecordSize;
    std::vector<uint8_t> hitGroupRecords(totalHitGroupRecordsArraySize);

    // ------------------------------------------------------------------
    // now, write all records (only on the host so far): we need to
    // write one record per geometry, per ray type
    // ------------------------------------------------------------------
    for (int groupID=0;groupID<groups.size();groupID++) {
      Group *group = groups.getPtr(groupID);
      if (!group) continue;
      GeomGroup *gg = dynamic_cast<GeomGroup *>(group);
      if (!gg) continue;
        
      const int sbtOffset = (int)gg->sbtOffset;
      PING; PRINT(groupID); PRINT(sbtOffset);
      for (int childID=0;childID<gg->geometries.size();childID++) {
        Geom::SP geom = gg->geometries[childID];
        if (!geom) continue;
          
        // const int geomID    = geom->ID;
        for (int rayTypeID=0;rayTypeID<numRayTypes;rayTypeID++) {
          // ------------------------------------------------------------------
          // compute pointer to entire record:
          // ------------------------------------------------------------------
          const int recordID
            = (sbtOffset+childID)*numRayTypes + rayTypeID;
          assert(recordID < numHitGroupRecords);
          uint8_t *const sbtRecord
            = hitGroupRecords.data() + recordID*hitGroupRecordSize;

          geom->writeSBTRecord(sbtRecord,device,rayTypeID);
          
          // // ------------------------------------------------------------------
          // // pack record header with the corresponding hit group:
          // // ------------------------------------------------------------------
          // // first, compute pointer to record:
          // char    *const sbtRecordHeader = (char *)sbtRecord;
          // // then, get gemetry we want to write (to find its hit group ID)...
          // // const Geom *const geom = checkGetGeom(geomID);
          // // ... find the PG that goes into the record header...
          
          // // auto geomType = geom->type;//device->geomTypes[geom->geomType->ID];
          // GeomType::DeviceData &gt = geom->type->getDD(device);
          // // const ll::HitGroupPG &hgPG
          // //   = geomType.perRayType[rayTypeID];
          // // ... and tell optix to write that into the record
          // OPTIX_CALL(SbtRecordPackHeader(gt.getPG(rayTypeID),sbtRecordHeader));
          
          // // ------------------------------------------------------------------
          // // finally, let the user fill in the record's payload using
          // // the callback
          // // ------------------------------------------------------------------
          // uint8_t *const sbtRecordData
          //   = sbtRecord + OPTIX_SBT_RECORD_HEADER_SIZE;
          // geom->writeVariables(sbtRecordData,device->ID);

          // std::cout << " writing geom " << geom->toString()
          //           << " raytype " << rayTypeID << " to offset " << recordID*hitGroupRecordSize << std::endl;
          // writeHitProgDataCB(sbtRecordData,
          //                    context->owlDeviceID,
          //                    geomID,
          //                    rayTypeID,
          //                    callBackUserData);
        }
      }
    }
    device->sbt.hitGroupRecordsBuffer.alloc(hitGroupRecords.size());
    device->sbt.hitGroupRecordsBuffer.upload(hitGroupRecords);
    PING; PRINT((void *)device->sbt.hitGroupRecordsBuffer.get());
    device->popActive(oldActive);
    LOG_OK("done building (and uploading) SBT hit group records");
  }
  

  void Context::buildMissProgRecordsOn(ll::Device *device)
  {
#if 1
    LOG("building SBT miss group records");
    int oldActive = device->pushActive();
    
    size_t numMissProgRecords = numRayTypes;
    size_t maxMissProgDataSize = 0;
    for (int i=0;i<missProgs.size();i++) {
      MissProg *missProg = (MissProg *)missProgs.getPtr(i);
      if (!missProg) continue;
        
      assert(missProg->type);
      maxMissProgDataSize = std::max(maxMissProgDataSize,missProg->type->varStructSize);
    }
    PING; PRINT(maxMissProgDataSize);
    
    size_t missProgRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxMissProgDataSize);
    PRINT(missProgRecordSize);
    
    assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    device->sbt.missProgRecordSize  = missProgRecordSize;
    device->sbt.missProgRecordCount = numMissProgRecords;

    size_t totalMissProgRecordsArraySize
      = numMissProgRecords * missProgRecordSize;
    std::vector<uint8_t> missProgRecords(totalMissProgRecordsArraySize);

    // ------------------------------------------------------------------
    // now, write all records (only on the host so far): we need to
    // write one record per geometry, per ray type
    // ------------------------------------------------------------------
    for (int recordID=0;recordID<numMissProgRecords;recordID++) {
      MissProg *miss
        = (recordID < missProgs.size())
        ? missProgs.getPtr(recordID)
        : nullptr;
      
      if (!miss) continue;
      
      uint8_t *const sbtRecord
        = missProgRecords.data() + recordID*missProgRecordSize;
      miss->writeSBTRecord(sbtRecord,device);
    }
    device->sbt.missProgRecordsBuffer.alloc(missProgRecords.size());
    device->sbt.missProgRecordsBuffer.upload(missProgRecords);
    device->popActive(oldActive);
    LOG_OK("done building (and uploading) SBT miss group records");
#else
    LOG("building SBT miss group records");
    
    size_t numMissProgRecords = missProgs.size();
    if (numMissProgRecords == 0) {
      std::cout << "warning: no miss progs!" << std::endl;
      return;
    }
    
    int oldActive = device->pushActive();

    size_t maxMissProgDataSize = 0;
    for (int i=0;i<missProgs.size();i++) {
      MissProg *missProg = (MissProg *)missProgs.getPtr(i);
      if (!missProg) continue;
        
      assert(missProg->type);
      maxMissProgDataSize = std::max(maxMissProgDataSize,missProg->type->varStructSize);
    }
    PING; PRINT(maxMissProgDataSize);
    
    PRINT(numRayTypes);
    PRINT(numMissProgRecords);
    assert(numMissProgRecords == numRayTypes);
    
    size_t missProgRecordSize
      = OPTIX_SBT_RECORD_HEADER_SIZE
      + smallestMultipleOf<OPTIX_SBT_RECORD_ALIGNMENT>(maxMissProgDataSize);
    PRINT(missProgRecordSize);
    
    assert((OPTIX_SBT_RECORD_HEADER_SIZE % OPTIX_SBT_RECORD_ALIGNMENT) == 0);
    device->sbt.missProgRecordSize  = missProgRecordSize;
    device->sbt.missProgRecordCount = numMissProgRecords;

    size_t totalMissProgRecordsArraySize
      = numMissProgRecords * missProgRecordSize;
    std::vector<uint8_t> missProgRecords(totalMissProgRecordsArraySize);

    // ------------------------------------------------------------------
    // now, write all records (only on the host so far): we need to
    // write one record per geometry, per ray type
    // ------------------------------------------------------------------
    for (int recordID=0;recordID<missProgs.size();recordID++) {
      MissProg *miss = missProgs.getPtr(recordID);
      if (!miss) continue;
        
      uint8_t *const sbtRecord
        = missProgRecords.data() + recordID*missProgRecordSize;
      miss->writeSBTRecord(sbtRecord,device);
    }
    device->sbt.missProgRecordsBuffer.alloc(missProgRecords.size());
    device->sbt.missProgRecordsBuffer.upload(missProgRecords);
    device->popActive(oldActive);
    LOG_OK("done building (and uploading) SBT miss group records");
#endif
  }


  void Context::buildRayGenRecordsOn(ll::Device *device)
  {
    LOG("building SBT rayGen prog records");
    int oldActive = device->pushActive();

    for (int rgID=0;rgID<rayGens.size();rgID++) {
      auto rg = rayGens.getPtr(rgID);
      assert(rg);
      auto &dd = rg->getDD(device);
      
      std::vector<uint8_t> hostMem(dd.rayGenRecordSize);
      rg->writeSBTRecord(hostMem.data(),device);
      dd.sbtRecordBuffer.upload(hostMem);
    }
    device->popActive(oldActive);
  }
  


  void Context::buildSBT(OWLBuildSBTFlags flags)
  {
// #if 1
    if (flags & OWL_SBT_HITGROUPS)
      for (auto device : llo->devices)
        buildHitGroupRecordsOn(device);
// #else
//     // ----------- build hitgroups -----------
//     if (flags & OWL_SBT_HITGROUPS)
//       llo->sbtHitProgsBuild
//         ([&](uint8_t *output,int devID,int geomID,int /*ignore: rayID*/) {
//           const Geom *geom = geoms.getPtr(geomID);
//           assert(geom);
//           geom->writeVariables(output,devID);
//         });
// #endif
    
    // ----------- build miss prog(s) -----------
    if (flags & OWL_SBT_MISSPROGS)
      for (auto device : llo->devices)
        buildMissProgRecordsOn(device);
      // llo->sbtMissProgsBuild
      //   ([&](uint8_t *output,
      //        int devID,
      //        int rayTypeID) {
      //     // TODO: eventually, we want to be able to 'assign' miss progs
      //     // to different ray types, in which case we ahve to be able to
      //     // look up wich miss prog is used for a given ray types - for
      //     // now, we assume miss progs are created in exactly the right
      //     // order ...
      //     int missProgID = rayTypeID;
      //     const MissProg *missProg = missProgs.getPtr(missProgID);
      //     assert(missProg);
      //     missProg->writeVariables(output,devID);
      //   });

    // ----------- build raygens -----------
    if (flags & OWL_SBT_RAYGENS)
      for (auto device : llo->devices)
        buildRayGenRecordsOn(device);
      // llo->sbtRayGensBuild
      //   ([&](uint8_t *output,
      //        int devID,
      //        int rgID) {

      //     // TODO: need the ID of the miss prog we're writing!
      //     int rayGenID = rgID;
      //     assert(rayGens.size() >= 1);
         
      //     const RayGen *rayGen = rayGens.getPtr(rayGenID);
      //     assert(rayGen);
      //     rayGen->writeVariables(output,devID);
      //   });
  
  }

  void Context::buildPipeline()
  {
    for (auto device : getDevices()) {
      getDD(device).destroyPipeline();
      getDD(device).buildPipeline();
    }
  }

  void Context::DeviceData::destroyPipeline()
  {
    if (!device->context->pipeline) return;
    
    const int oldActive = device->pushActive();
    
    OPTIX_CHECK(optixPipelineDestroy(device->context->pipeline));
    device->context->pipeline = 0;
    
    device->popActive(oldActive);
  }
  
  void Context::DeviceData::buildPipeline()
  {
    const int oldActive = device->pushActive();
    
    auto &allPGs = allActivePrograms;
    if (allPGs.empty())
      throw std::runtime_error("trying to create a pipeline w/ 0 programs!?");
      
    char log[2048];
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK(optixPipelineCreate(device->context->optixContext,
                                    &device->context->pipelineCompileOptions,
                                    &device->context->pipelineLinkOptions,
                                    allPGs.data(),
                                    (uint32_t)allPGs.size(),
                                    log,&sizeof_log,
                                    &device->context->pipeline
                                    ));
      
    uint32_t maxAllowedByOptix = 0;
    optixDeviceContextGetProperty
      (device->context->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
       &maxAllowedByOptix,
       sizeof(maxAllowedByOptix));
    if (uint32_t(parent->maxInstancingDepth+1) > maxAllowedByOptix)
      throw std::runtime_error
        ("error when building pipeline: "
         "attempting to set max instancing depth to "
         "value that exceeds OptiX's MAX_TRAVERSABLE_GRAPH_DEPTH limit");

    PRINT(parent->maxInstancingDepth);
    PRINT(maxAllowedByOptix);
    
    OPTIX_CHECK(optixPipelineSetStackSize
                (device->context->pipeline,
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

    device->popActive(oldActive);
  }

  void Context::buildModules()
  {
    destroyModules();
    for (auto device : getDevices()) {
      device->context->configurePipelineOptions(this);
      for (int moduleID=0;moduleID<modules.size();moduleID++) {
        Module *module = modules.getPtr(moduleID);
        if (!module) continue;
        
        module->getDD(device).build(module,device);
      }
    }
  }
  
  void Context::setRayTypeCount(size_t rayTypeCount)
  {
    /* TODO; sanity checking that this is a useful value, and that
       no geoms etc are created yet */
    this->numRayTypes = rayTypeCount;
      
    // lloSetRayTypeCount(llo,rayTypeCount);
    // llo->setRayTypeCount(rayTypeCount);
  }

  /*! sets maximum instancing depth for the given context:

    '0' means 'no instancing allowed, only bottom-level accels; 
  
    '1' means 'at most one layer of instances' (ie, a two-level scene),
    where the 'root' world rays are traced against can be an instance
    group, but every child in that inscne group is a geometry group.

    'N>1" means "up to N layers of instances are allowed.

    The default instancing depth is 1 (ie, a two-level scene), since
    this allows for most use cases of instancing and is still
    hardware-accelerated. Using a node graph with instancing deeper than
    the configured value will result in wrong results; but be aware that
    using any value > 1 here will come with a cost. It is recommended
    to, if at all possible, leave this value to one and convert the
    input scene to a two-level scene layout (ie, with only one level of
    instances) */
  void Context::setMaxInstancingDepth(int32_t maxInstanceDepth)
  {
    this->maxInstancingDepth = maxInstanceDepth;
    
    if (maxInstancingDepth < 1)
      throw std::runtime_error("a instancing depth of < 1 isnt' currently supported in OWL; pleaes see comments on owlSetMaxInstancingDepth() (owl/owl_host.h)");
    
    for (auto device : llo->devices) {
      assert("check pipeline isn't already created"
             && device->context->pipeline == nullptr);
      // device->context->maxInstancingDepth = maxInstancingDepth;
      device->context->configurePipelineOptions(this);
    } 
  }

  void Context::enableMotionBlur()
  {
    motionBlurEnabled = true;
    // // todo: axe this
    // llo->enableMotionBlur();
  }

  void Context::buildPrograms()
  {
    buildModules();
    
    for (auto device : getDevices()) {
      const int oldActive = device->pushActive();
      auto &dd = getDD(device);
      dd.buildPrograms();
      device->popActive(oldActive);
    }
  }


  void Context::destroyModules()
  {
    for (int moduleID=0;moduleID<modules.size();moduleID++) {
      Module *module = modules.getPtr(moduleID);
      if (module)
        for (auto device : getDevices())
          module->getDD(device).destroy(device);
    }
  }
    
  void Context::DeviceData::destroyPrograms()
  {
    const int oldActive = device->pushActive();
    for (auto pg : allActivePrograms)
      optixProgramGroupDestroy(pg);
    allActivePrograms.clear();
    device->popActive(oldActive);
  }

  void Context::destroyPrograms()
  {
    for (auto device : getDevices()) 
      getDD(device).destroyPrograms();
  }

  void Context::DeviceData::buildMissPrograms()
  {
    // ------------------------------------------------------------------
    // miss programs
    // ------------------------------------------------------------------
    for (int progID=0;progID<parent->missProgTypes.size();progID++) {
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc    pgDesc    = {};
      
      MissProgType *prog = parent->missProgTypes.getPtr(progID);
      if (!prog) continue;
      auto &dd = prog->getDD(device);
      assert(dd.pg == 0);

      Module::SP module = prog->module;
      assert(module);
      
      pgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
      OptixModule optixModule = module->getDD(device).module;
      assert(optixModule);
      
      const std::string annotatedProgName
        = std::string("__miss__")+prog->progName;
      pgDesc.miss.module            = optixModule;
      pgDesc.miss.entryFunctionName = annotatedProgName.c_str();
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(device->context->optixContext,
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
  
  void Context::DeviceData::buildRayGenPrograms()
  {
    // ------------------------------------------------------------------
    // rayGen programs
    // ------------------------------------------------------------------

    for (int pgID=0;pgID<parent->rayGenTypes.size();pgID++) {
      OptixProgramGroupOptions pgOptions = {};
      OptixProgramGroupDesc    pgDesc    = {};
      
      RayGenType *prog = parent->rayGenTypes.getPtr(pgID);
      if (!prog) continue;
      
      auto &dd = prog->getDD(device);
      assert(dd.pg == 0);
      
      Module::SP module = prog->module;
      assert(module);
      
      OptixModule optixModule = module->getDD(device).module;
      assert(optixModule);
      
      pgDesc.kind                     = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
      const std::string annotatedProgName
        = std::string("__raygen__")+prog->progName;
      pgDesc.raygen.module            = optixModule;
      pgDesc.raygen.entryFunctionName = annotatedProgName.c_str();
      
      char log[2048];
      size_t sizeof_log = sizeof( log );
      OPTIX_CHECK(optixProgramGroupCreate(device->context->optixContext,
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
  
  void Context::DeviceData::buildHitGroupPrograms()
  {
    const int numRayTypes = parent->numRayTypes;
    
    // ------------------------------------------------------------------
    // geometry type programs -> what goes into hit groups
    // ------------------------------------------------------------------
    for (int geomTypeID=0;geomTypeID<parent->geomTypes.size();geomTypeID++) {
      GeomType::SP geomType = parent->geomTypes.getSP(geomTypeID);
      if (!geomType)
        continue;

      UserGeomType::SP userGeomType
        = geomType->as<UserGeomType>();
      if (userGeomType)
        userGeomType->buildBoundsProg();
                          
      auto &dd = geomType->getDD(device);
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
        dd.fillPGDesc(pgDesc,geomType.get(),device,rt);
        
        char log[2048];
        size_t sizeof_log = sizeof( log );
        OptixProgramGroup &pg = dd.hgPGs[rt];
        OPTIX_CHECK(optixProgramGroupCreate(device->context->optixContext,
                                            &pgDesc,
                                            1,
                                            &pgOptions,
                                            log,&sizeof_log,
                                            &pg
                                            ));
        allActivePrograms.push_back(pg);
      }
    }
  }
  
  // void Context::DeviceData::buildBoundsPrograms()
  // {
  //   throw std::runtime_error("not implemented");
  // }
  
  void Context::DeviceData::buildPrograms()
  {
    int oldActive = device->pushActive();
    destroyPrograms();
    buildMissPrograms();
    buildRayGenPrograms();
    buildHitGroupPrograms();
    device->popActive(oldActive);
  }

#if 0
  void Context::buildOptixPrograms(Device *device)
  {
    int oldActive = context->pushActive();
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
    context->popActive(oldActive);
  }
#endif
  
  // void Context::destroyPrograms(Device *device)
  // {
  //   for (auto type : rayGenTypes.objects)
  //     type->destroyProgramGroups(device);
  //   for (auto type : geomTypes.objects)
  //     type->destroyProgramGroups(device);
  //   for (auto type : missProgTypes.objects)
  //     type->destroyProgramGroups(device);
    
  //   // int oldActive = device->pushActive();
    
  //   // // ---------------------- rayGen ----------------------
  //   // // for (auto &pg : rayGenPGs) {
  //   // //   if (pg.pg) optixProgramGroupDestroy(pg.pg);
  //   // //   pg.pg = nullptr;
  //   // // }
  //   // for (auto rg : rayGens.objects) rg->destroyProgramGroups(device);
    
  //   // // ---------------------- hitGroup ----------------------
  //   // for (auto &geomType : geomTypes) 
  //   //   for (auto &pg : geomType.perRayType) {
  //   //     if (pg.pg) optixProgramGroupDestroy(pg.pg);
  //   //     pg.pg = nullptr;
  //   //   }
  //   // // ---------------------- miss ----------------------
  //   // for (auto &pg : missProgPGs) {
  //   //   if (pg.pg) optixProgramGroupDestroy(pg.pg);
  //   //   pg.pg = nullptr;
  //   // }

  //   // device->popActive(oldActive);
  // }
  
} // ::owl
