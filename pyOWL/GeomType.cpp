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

#include "pyOWL/Context.h"

namespace pyOWL {

  GeomType::GeomType(Context *ctx,
                     OWLGeomKind kind,
                     const std::shared_ptr<Module> &module,
                     const std::string &typeName)
    : module(module),
      typeName(typeName),
      kind(kind)
  {
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "#pyOWL: creating GeomType " << typeName
              << OWL_TERMINAL_DEFAULT
              << std::endl;

    size_t typeSize
      = module->getTypeSize(typeName);
    std::vector<OWLVarDecl> vars
      = module->getTypeVars(typeName);
    if (vars.empty())
      std::cout << OWL_TERMINAL_RED
                << "#pyOWL: warning: type " << typeName
                << " has no variables in this module!?"
                << OWL_TERMINAL_DEFAULT
                << std::endl;

    handle = owlGeomTypeCreate(ctx->handle,
                               kind,
                               typeSize,
                               vars.data(),
                               vars.size());
    assert(handle);
    
    std::cout << OWL_TERMINAL_GREEN
              << "#pyOWL: GeomType created."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
  }
    
  void GeomType::setClosestHit(int rayType,
                               const Module::SP &module,
                               const std::string &fctName)
  {
    owlGeomTypeSetClosestHit(handle,rayType,module->getHandle(),fctName.c_str());
  }

  void GeomType::setIntersectProg(int rayType,
                                  const Module::SP &module,
                                  const std::string &fctName)
  {
    owlGeomTypeSetIntersectProg(handle,rayType,module->getHandle(),fctName.c_str());
  }
  
  void GeomType::setBoundsProg(const Module::SP &module,
                               const std::string &fctName)
  {
    owlGeomTypeSetBoundsProg(handle,module->getHandle(),fctName.c_str());
  }
  
}
