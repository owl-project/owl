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
    LOG_OK("device group created");
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
    buffer->createDeviceData(llo->devices);
    buffer->resize(count);

    return buffer;
  }

  RayGen::SP
  Context::createRayGen(const std::shared_ptr<RayGenType> &type)
  {
    return std::make_shared<RayGen>(this,type);
  }

  LaunchParams::SP
  Context::createLaunchParams(const std::shared_ptr<LaunchParamsType> &type)
  {
    return std::make_shared<LaunchParams>(this,type);
  }

  MissProg::SP
  Context::createMissProg(const std::shared_ptr<MissProgType> &type)
  {
    return std::make_shared<MissProg>(this,type);
  }

  RayGenType::SP
  Context::createRayGenType(Module::SP module,
                            const std::string &progName,
                            size_t varStructSize,
                            const std::vector<OWLVarDecl> &varDecls)
  {
    return std::make_shared<RayGenType>(this,
                                        module,progName,
                                        varStructSize,
                                        varDecls);
  }
  

  LaunchParamsType::SP
  Context::createLaunchParamsType(size_t varStructSize,
                                  const std::vector<OWLVarDecl> &varDecls)
  {
    return std::make_shared<LaunchParamsType>(this,
                                              varStructSize,
                                              varDecls);
  }
  

  MissProgType::SP
  Context::createMissProgType(Module::SP module,
                              const std::string &progName,
                              size_t varStructSize,
                              const std::vector<OWLVarDecl> &varDecls)
  {
    return std::make_shared<MissProgType>(this,
                                          module,progName,
                                          varStructSize,
                                          varDecls);
  }
  

  GeomGroup::SP Context::trianglesGeomGroupCreate(size_t numChildren)
  {
    return std::make_shared<TrianglesGeomGroup>(this,numChildren);
  }

  GeomGroup::SP Context::userGeomGroupCreate(size_t numChildren)
  {
    return std::make_shared<UserGeomGroup>(this,numChildren);
  }


  GeomType::SP
  Context::createGeomType(OWLGeomKind kind,
                          size_t varStructSize,
                          const std::vector<OWLVarDecl> &varDecls)
  {
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      return std::make_shared<TrianglesGeomType>(this,varStructSize,varDecls);
    case OWL_GEOMETRY_USER:
      return std::make_shared<UserGeomType>(this,varStructSize,varDecls);
    default:
      OWL_NOTIMPLEMENTED;
    }
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    return std::make_shared<Module>(this,ptxCode);//,modules.allocID());
  }

  std::shared_ptr<Geom> UserGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    assert(self);

    return std::make_shared<UserGeom>(context,self);
  }

  std::shared_ptr<Geom> TrianglesGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    assert(self);
    return std::make_shared<TrianglesGeom>(context,self);
  }

  void Context::buildHitGroupsOn(ll::Device *device)
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
    size_t numHitGroupRecords = numHitGroupEntries*numRayTypes;
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
  
  void Context::buildSBT(OWLBuildSBTFlags flags)
  {
#if 1
    if (flags & OWL_SBT_HITGROUPS)
      for (auto device : llo->devices)
        buildHitGroupsOn(device);
#else
    // ----------- build hitgroups -----------
    if (flags & OWL_SBT_HITGROUPS)
      llo->sbtHitProgsBuild
        ([&](uint8_t *output,int devID,int geomID,int /*ignore: rayID*/) {
          const Geom *geom = geoms.getPtr(geomID);
          assert(geom);
          geom->writeVariables(output,devID);
        });
#endif
    
    // ----------- build miss prog(s) -----------
    if (flags & OWL_SBT_MISSPROGS)
      llo->sbtMissProgsBuild
        ([&](uint8_t *output,
             int devID,
             int rayTypeID) {
          // TODO: eventually, we want to be able to 'assign' miss progs
          // to different ray types, in which case we ahve to be able to
          // look up wich miss prog is used for a given ray types - for
          // now, we assume miss progs are created in exactly the right
          // order ...
          int missProgID = rayTypeID;
          const MissProg *missProg = missProgs.getPtr(missProgID);
          assert(missProg);
          missProg->writeVariables(output,devID);
        });

    // ----------- build raygens -----------
    if (flags & OWL_SBT_RAYGENS)
      llo->sbtRayGensBuild
        ([&](uint8_t *output,
             int devID,
             int rgID) {

          // TODO: need the ID of the miss prog we're writing!
          int rayGenID = rgID;
          assert(rayGens.size() >= 1);
         
          const RayGen *rayGen = rayGens.getPtr(rayGenID);
          assert(rayGen);
          rayGen->writeVariables(output,devID);
        });
  
  }

  void Context::buildPipeline()
  {
    throw std::runtime_error("not implemented");
    // lloCreatePipeline(llo);
    // llo->createPipeline();
  }

  void Context::buildModules()
  {
    throw std::runtime_error("not implemented");
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
    maxInstanceDepth = maxInstanceDepth;
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
    for (auto device : getDevices())
      buildPrograms(device);
  }

  void Context::destroyPrograms()
  {
    for (auto device : getDevices())
      destroyPrograms(device);
  }


  // void Context::buildPrograms(Device *device)
  // {
  //   buildMissPrograms(device);
  //   buildRayGenPrograms(device);
  //   buildBoundsPrograms(device);
  //   buildIsecPrograms(device);
  //   buildCHPrograms(device);
  //   buildAHPrograms(device);
  // }

  
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
    
  void Context::destroyPrograms(Device *device)
  {
    for (auto type : rayGenTypes.objects)
      type->destroyProgramGroups(device);
    for (auto type : geomTypes.objects)
      type->destroyProgramGroups(device);
    for (auto type : missProgTypes.objects)
      type->destroyProgramGroups(device);
    
    // int oldActive = device->pushActive();
    
    // // ---------------------- rayGen ----------------------
    // // for (auto &pg : rayGenPGs) {
    // //   if (pg.pg) optixProgramGroupDestroy(pg.pg);
    // //   pg.pg = nullptr;
    // // }
    // for (auto rg : rayGens.objects) rg->destroyProgramGroups(device);
    
    // // ---------------------- hitGroup ----------------------
    // for (auto &geomType : geomTypes) 
    //   for (auto &pg : geomType.perRayType) {
    //     if (pg.pg) optixProgramGroupDestroy(pg.pg);
    //     pg.pg = nullptr;
    //   }
    // // ---------------------- miss ----------------------
    // for (auto &pg : missProgPGs) {
    //   if (pg.pg) optixProgramGroupDestroy(pg.pg);
    //   pg.pg = nullptr;
    // }

    // device->popActive(oldActive);
  }
  
} // ::owl
