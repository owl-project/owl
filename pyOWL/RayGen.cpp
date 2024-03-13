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

#include "pyOWL/RayGen.h"
#include "pyOWL/Context.h"

namespace pyOWL {

  RayGen::SP RayGen::create(Context *ctx,
                            const Module::SP &module,
                            const std::string &typeName,
                            const std::string &funcName)
  {
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "#pyOWL: creating raygen prog..." << typeName
              << OWL_TERMINAL_DEFAULT
              << std::endl;

    size_t typeSize
      = module->getTypeSize(typeName);
    std::vector<OWLVarDecl> vars
      = module->getTypeVars(typeName);

    OWLRayGen handle
      = owlRayGenCreate(ctx->handle,
                        module->getHandle(),
                        funcName.c_str(),
                        typeSize,
                        vars.data(),
                        vars.size());
    
    std::cout << OWL_TERMINAL_GREEN
              << "#pyOWL: ray gen "
              << typeName << "::" << funcName << " successfully created."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    
    return std::make_shared<RayGen>(handle);
  }
  
}
