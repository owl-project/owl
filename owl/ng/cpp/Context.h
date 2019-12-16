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

#include "ObjectRegistry.h"
#include "Buffer.h"
#include "Group.h"
#include "RayGen.h"
#include "MissProg.h"
// ll
#include "ll/llowl.h"

namespace owl {

  std::string typeToString(const OWLDataType type);
  
  struct Context : public Object {
    typedef std::shared_ptr<Context> SP;

    static Context::SP create();

    Context();
    
    virtual ~Context()
    {
      std::cout << "=======================================================" << std::endl;
      std::cout << "#owl: destroying context" << std::endl;
    }

    ObjectRegistryT<Buffer>       buffers;
    ObjectRegistryT<Group>        groups;
    ObjectRegistryT<RayGenType>   rayGenTypes;
    ObjectRegistryT<RayGen>       rayGens;
    ObjectRegistryT<MissProgType> missProgTypes;
    ObjectRegistryT<MissProg>     missProgs;
    ObjectRegistryT<GeomType>     geomTypes;
    ObjectRegistryT<Geom>         geoms;
    ObjectRegistryT<Module>       modules;
    
    //! TODO: allow changing that via api ..
    size_t numRayTypes = 1;

    /*! experimentation code for sbt construction */
    void expBuildSBT();
    
    InstanceGroup::SP
    createInstanceGroup(size_t numChildren);
    
    GeomGroup::SP
    trianglesGroupCreate(size_t numChildren);
    
    Buffer::SP
    deviceBufferCreate();
    Buffer::SP
    hostPinnedBufferCreate();

    RayGen::SP
    createRayGen(const std::shared_ptr<RayGenType> &type);
    
    RayGenType::SP
    createRayGenType(Module::SP module,
                     const std::string &progName,
                     size_t varStructSize,
                     const std::vector<OWLVarDecl> &varDecls);
    
    MissProg::SP
    createMissProg(const std::shared_ptr<MissProgType> &type);
    
    MissProgType::SP
    createMissProgType(Module::SP module,
                       const std::string &progName,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);
    
    GeomType::SP
    createGeomType(OWLGeomKind kind,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);
    
    Module::SP createModule(const std::string &ptxCode);

    ll::DeviceGroup::SP ll;
  };

} // ::owl

