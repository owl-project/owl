// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include "pyOWL/common.h"
#include "pyOWL/Module.h"

namespace pyOWL {

  struct Context;
  
  struct GeomType : public std::enable_shared_from_this<GeomType>
  {
    typedef std::shared_ptr<GeomType> SP;

    GeomType(Context *ctx,
             OWLGeomKind kind,
             const Module::SP &module,
             const std::string &name);
    
    void setClosestHit(int rayType,
                       const Module::SP &module,
                       const std::string &fctName);

    void setIntersectProg(int rayType,
                          const Module::SP &module,
                          const std::string &fctName);
    
    void setBoundsProg(const Module::SP &module,
                       const std::string &fctName);
    
    const std::string typeName;
    const OWLGeomKind kind;
    const std::shared_ptr<Module> module;
    OWLGeomType handle = 0;
  };
  
}
