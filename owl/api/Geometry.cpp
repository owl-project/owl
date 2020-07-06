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
    context->llo->geomTypeCreate(this->ID,
                                 varStructSize);
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
      
    closestHit[rayType].progName = progName;
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
      
    anyHit[rayType].progName = progName;
    anyHit[rayType].module   = module;
    // context->llo->setGeomTypeAnyHit(this->ID,
    //                       rayType,module->ID,
    //                       // warning: this 'this' here is importat, since
    //                       // *we* manage the lifetime of this string, and
    //                       // the one on the constructor list will go out of
    //                       // scope after this function
    //                       anyHit[rayType].progName.c_str());
  }

          
  void GeomType::writeSBTHeader(uint8_t *const sbtRecord,
                                Device *device,
                                int rayTypeID)
  {
    // // auto geomType = geom->type;//device->geomTypes[geom->geomType->ID];
    // GeomType::DeviceData &gt = geom->type->getDD(device);
    // // const ll::HitGroupPG &hgPG
    // //   = geomType.perRayType[rayTypeID];
    // // ... and tell optix to write that into the record
    // OPTIX_CALL(SbtRecordPackHeader(gt.getPG(rayTypeID),sbtRecordHeader));
    throw std::runtime_error("not implemented");
  }

  void Geom::writeSBTRecord(uint8_t *const sbtRecord,
                            Device *device,
                            int rayTypeID)
  {
    // first, compute pointer to record:
    uint8_t *const sbtRecordHeader = sbtRecord;
    uint8_t *const sbtRecordData   = sbtRecord+OPTIX_SBT_RECORD_HEADER_SIZE;

    // ------------------------------------------------------------------
    // pack record header with the corresponding hit group:
    // ------------------------------------------------------------------
    geomType->getDD(device).writeSBTHeader(sbtRecordHeader,
                                           device,rayTypeID);
    
    // ------------------------------------------------------------------
    // then, write the data for that record
    // ------------------------------------------------------------------
    writeVariables(sbtRecordData,device->ID);
  }  

} //::owl
