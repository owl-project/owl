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

#include "LaunchParams.h"
#include "Context.h"

namespace owl {

  LaunchParamsType::LaunchParamsType(Context *const context,
                                     size_t varStructSize,
                                     const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->launchParamTypes,varStructSize,varDecls)
  {
  }
  
  LaunchParams::LaunchParams(Context *const context,
                 LaunchParamsType::SP type) 
    : SBTObject(context,context->launchParams,type)
  {
    assert(context);
    assert(type);
    assert(type.get());
    context->llo->launchParamsCreate(this->ID,
                                     type->varStructSize);
  }

  CUstream LaunchParams::getCudaStream(int deviceID)
  {
    return context->llo->launchParamsGetStream(this->ID,deviceID);
  }

} // ::owl

