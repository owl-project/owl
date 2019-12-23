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

#include "RayGen.h"
#include "Context.h"

namespace owl {

  RayGenType::RayGenType(Context *const context,
                         Module::SP module,
                         const std::string &progName,
                         size_t varStructSize,
                         const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->rayGenTypes,varStructSize,varDecls),
      module(module),
      progName(progName)
  {
    PING; PRINT(this);
  }
  
  RayGen::RayGen(Context *const context,
                 RayGenType::SP type) 
    : SBTObject(context,context->rayGens,type)
  {
    PING;
    PRINT(type.get());
    PRINT(type->progName);
    assert(context);
    assert(type);
    assert(type.get());
    assert(type->module);
    assert(type->progName != "");
    lloRayGenCreate(context->llo,this->ID,
                    type->module->ID,
                    type->progName.c_str(),
                    type->varStructSize);
  }
  
} // ::owl

