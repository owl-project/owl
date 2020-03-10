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

#pragma once

#include "RegisteredObject.h"
#include "SBTObject.h"
#include "Module.h"
#include "Buffer.h"

namespace owl {

  struct Geom;

  struct ProgramDesc {
    Module::SP  module;
    std::string progName;
  };
  
  struct GeomType : public SBTObjectType {
    typedef std::shared_ptr<GeomType> SP;
    
    GeomType(Context *const context,
             size_t varStructSize,
             const std::vector<OWLVarDecl> &varDecls);
    
    virtual std::string toString() const { return "GeomType"; }
    virtual void setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName);

    std::vector<ProgramDesc> closestHit;
    
    virtual void setAnyHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName);

    std::vector<ProgramDesc> anyHit;
    virtual std::shared_ptr<Geom> createGeom() = 0;
  };

  struct TrianglesGeomType : public GeomType {
    typedef std::shared_ptr<TrianglesGeomType> SP;
    
    TrianglesGeomType(Context *const context,
                      size_t varStructSize,
                      const std::vector<OWLVarDecl> &varDecls);

    virtual std::string toString() const { return "TriangleGeomType"; }
    virtual std::shared_ptr<Geom> createGeom() override;
  };

  struct UserGeomType : public GeomType {
    typedef std::shared_ptr<UserGeomType> SP;
    
    UserGeomType(Context *const context,
                 size_t varStructSize,
                 const std::vector<OWLVarDecl> &varDecls);

    virtual void setIntersectProg(int rayType,
                                  Module::SP module,
                                  const std::string &progName);
    virtual void setBoundsProg(Module::SP module,
                               const std::string &progName);
    
    virtual std::string toString() const { return "UserGeomType"; }
    virtual std::shared_ptr<Geom> createGeom() override;

    ProgramDesc boundsProg;
    std::vector<ProgramDesc> intersectProg;
  };
  
  struct Geom : public SBTObject<GeomType> {
    typedef std::shared_ptr<Geom> SP;

    Geom(Context *const context,
             GeomType::SP geometryType);
    virtual std::string toString() const { return "Geom"; }
    
    GeomType::SP geometryType;
  };

  struct TrianglesGeom : public Geom {
    typedef std::shared_ptr<TrianglesGeom> SP;

    TrianglesGeom(Context *const context,
                  GeomType::SP geometryType);
    
    void setVertices(Buffer::SP vertices,
                     size_t count,
                     size_t stride,
                     size_t offset);
    void setIndices(Buffer::SP indices,
                    size_t count,
                    size_t stride,
                    size_t offset);
    virtual std::string toString() const { return "TrianglesGeom"; }
  };

  struct UserGeom : public Geom {
    typedef std::shared_ptr<UserGeom> SP;

    UserGeom(Context *const context,
             GeomType::SP geometryType);

    virtual std::string toString() const { return "UserGeom"; }
    void setPrimCount(size_t count);
  };
  
} // ::owl
