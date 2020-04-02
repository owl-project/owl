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
    : SBTObject(context,context->geoms,geomType)
  {
    assert(geomType);
  }





  TrianglesGeomType::TrianglesGeomType(Context *const context,
                                       size_t varStructSize,
                                       const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {
    /*! nothing special - all inherited */
  }

  UserGeomType::UserGeomType(Context *const context,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls),
      intersectProg(context->numRayTypes)
  {
    /*! nothing special - all inherited */
  }

  TrianglesGeom::TrianglesGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
  {
    context->llo->trianglesGeomCreate(this->ID,geometryType->ID);
  }

  UserGeom::UserGeom(Context *const context,
                     GeomType::SP geometryType)
    : Geom(context,geometryType)
  {
    int numPrims = 0;
    context->llo->userGeomCreate(this->ID,geometryType->ID,numPrims);
  }

  void UserGeom::setPrimCount(size_t count)
  {
    context->llo->userGeomSetPrimCount(this->ID,count);
  }




  void TrianglesGeom::setVertices(Buffer::SP vertices,
                                  size_t count,
                                  size_t stride,
                                  size_t offset)
  {
    context->llo->trianglesGeomSetVertexBuffer(this->ID,
                                    vertices->ID,count,stride,offset);
  }
  
  void TrianglesGeom::setIndices(Buffer::SP indices,
                                 size_t count,
                                 size_t stride,
                                 size_t offset)
  {
    context->llo->trianglesGeomSetIndexBuffer(this->ID,
                                    indices->ID, count, stride, offset);
  }

  void GeomType::setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < closestHit.size());
      
    closestHit[rayType].progName = progName;
    closestHit[rayType].module   = module;
    context->llo->setGeomTypeClosestHit(this->ID,
                          rayType,module->ID,
                          // warning: this 'this' here is importat, since
                          // *we* manage the lifetime of this string, and
                          // the one on the constructor list will go out of
                          // scope after this function
                          closestHit[rayType].progName.c_str());
  }

  void GeomType::setAnyHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < anyHit.size());
      
    anyHit[rayType].progName = progName;
    anyHit[rayType].module   = module;
    context->llo->setGeomTypeAnyHit(this->ID,
                          rayType,module->ID,
                          // warning: this 'this' here is importat, since
                          // *we* manage the lifetime of this string, and
                          // the one on the constructor list will go out of
                          // scope after this function
                          anyHit[rayType].progName.c_str());
  }


  void UserGeomType::setIntersectProg(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < intersectProg.size());
    intersectProg[rayType].progName = progName;
    intersectProg[rayType].module   = module;
    context->llo->setGeomTypeIntersect(this->ID,
                         rayType,module->ID,
                         // warning: this 'this' here is importat, since
                         // *we* manage the lifetime of this string, and
                         // the one on the constructor list will go out of
                          // scope after this function
                         intersectProg[rayType].progName.c_str());
  }

  void UserGeomType::setBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->boundsProg.progName = progName;
    this->boundsProg.module   = module;
    context->llo->setGeomTypeBoundsProgDevice(this->ID,
                                module->ID,
                                // warning: this 'this' here is importat, since
                                // *we* manage the lifetime of this string, and
                                // the one on the constructor list will go out of
                                // scope after this function
                                this->boundsProg.progName.c_str(),
                                varStructSize
                                );
  }

} //::owl
