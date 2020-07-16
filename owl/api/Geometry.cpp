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

#include "Geometry.h"
#include "Context.h"

namespace owl {
  
  GeomType::GeomType(Context *const context,
                     size_t varStructSize,
                     const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->geomTypes,
                    varStructSize,varDecls),
      closestHit(context->numRayTypes),
      anyHit(context->numRayTypes)
  {
    // context->llo->geomTypeCreate(this->ID,
    //                              varStructSize);
  }
  
  Geom::Geom(Context *const context,
             GeomType::SP geomType)
    : SBTObject(context,context->geoms,geomType), geomType(geomType)
  {
    assert(geomType);
  }



  void GeomType::setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < closestHit.size());
      
    closestHit[rayType].progName = "__closesthit__"+progName;
    closestHit[rayType].module   = module;
    // context->llo->setGeomTypeClosestHit(this->ID,
    //                       rayType,module->ID,
    //                       // warning: this 'this' here is importat, since
    //                       // *we* manage the lifetime of this string, and
    //                       // the one on the constructor list will go out of
    //                       // scope after this function
    //                       closestHit[rayType].progName.c_str());
  }

  void GeomType::setAnyHitProgram(int rayType,
                                  Module::SP module,
                                  const std::string &progName)
  {
    assert(rayType < anyHit.size());
      
    anyHit[rayType].progName = "__anyhit__"+progName;
    anyHit[rayType].module   = module;
    // context->llo->setGeomTypeAnyHit(this->ID,
    //                       rayType,module->ID,
    //                       // warning: this 'this' here is importat, since
    //                       // *we* manage the lifetime of this string, and
    //                       // the one on the constructor list will go out of
    //                       // scope after this function
    //                       anyHit[rayType].progName.c_str());
  }

          
  // void GeomType::DeviceData::writeSBTHeader(uint8_t *const sbtRecord,
  //                                           Device *device,
  //                                           int rayTypeID)
  // {
    
  //   // // auto geomType = geom->type;//device->geomTypes[geom->geomType->ID];
  //   // GeomType::DeviceData &gt = geom->type->getDD(device);
  //   // // const ll::HitGroupPG &hgPG
  //   // //   = geomType.perRayType[rayTypeID];
  //   // // ... and tell optix to write that into the record
  //   // OPTIX_CALL(SbtRecordPackHeader(gt.getPG(rayTypeID),sbtRecordHeader));
  //   // throw std::runtime_error("not implemented");
  // }

  void Geom::writeSBTRecord(uint8_t *const sbtRecord,
                            const DeviceContext::SP &device,
                            int rayTypeID)
  {
    // first, compute pointer to record:
    uint8_t *const sbtRecordHeader = sbtRecord;
    uint8_t *const sbtRecordData   = sbtRecord+OPTIX_SBT_RECORD_HEADER_SIZE;

    // ------------------------------------------------------------------
    // pack record header with the corresponding hit group:
    // ------------------------------------------------------------------
    auto &dd = geomType->getDD(device);
    assert(rayTypeID < dd.hgPGs.size());
    OPTIX_CALL(SbtRecordPackHeader(dd.hgPGs[rayTypeID],sbtRecordHeader));
    
    // ------------------------------------------------------------------
    // then, write the data for that record
    // ------------------------------------------------------------------
    writeVariables(sbtRecordData,device);
  }  




  
  void GeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                        GeomType *parent,
                                        int rt)
  {
    // ----------- closest hit -----------
    if (rt < parent->closestHit.size()) {
      const ProgramDesc &pd = parent->closestHit[rt];
      if (pd.module) {
        pgDesc.hitgroup.moduleCH = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameCH = pd.progName.c_str();
      }
    }
    // ----------- any hit -----------
    if (rt < parent->anyHit.size()) {
      const ProgramDesc &pd = parent->anyHit[rt];
      if (pd.module) {
        std::string annotatedProgName
          = std::string("__anyhit__")+pd.progName;
        pgDesc.hitgroup.moduleAH = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameAH = annotatedProgName.c_str();
      }
    }
  }
  

  // void GeomType::DeviceData::buildHitGroupPrograms(GeomType *gt,
  //                                                  Device *device)
  // {
  //   const int numRayTypes = gt->context->numRayTypes;
  //   hgPGs.resize(numRayTypes);
        
  //   for (int rt=0;rt<numRayTypes;rt++) {
      
  //     OptixProgramGroupOptions pgOptions = {};
  //     OptixProgramGroupDesc    pgDesc    = {};
      
  //     pgDesc.kind      = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
  //     // ----------- default init closesthit -----------
  //     pgDesc.hitgroup.moduleCH            = nullptr;
  //     pgDesc.hitgroup.entryFunctionNameCH = nullptr;
  //     // ----------- default init anyhit -----------
  //     pgDesc.hitgroup.moduleAH            = nullptr;
  //     pgDesc.hitgroup.entryFunctionNameAH = nullptr;
  //     // ----------- default init intersect -----------
  //     pgDesc.hitgroup.moduleIS            = nullptr;
  //     pgDesc.hitgroup.entryFunctionNameIS = nullptr;

  //     // now let the type fill in what it has
  //     fillPGDesc(pgDesc,gt,device,rt);
        
  //     char log[2048];
  //     size_t sizeof_log = sizeof( log );
  //     OPTIX_CHECK(optixProgramGroupCreate(device->context->optixContext,
  //                                         &pgDesc,
  //                                         1,
  //                                         &pgOptions,
  //                                         log,&sizeof_log,
  //                                         &pg
  //                                         ));
  //     allActivePrograms.push_back(pg);
  //   }

} //::owl
