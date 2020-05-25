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

#include "Module.h"
#include "Context.h"

namespace owl {

  Module::Module(Context *const context,
                 const std::string &ptxCode)
    : RegisteredObject(context,context->modules),
      ptxCode(ptxCode)
  {
    // lloModuleCreate(context->llo,this->ID,
    context->llo->moduleCreate(this->ID,
                               // warning: this 'this' here is importat, since
                               // *we* manage the lifetime of this string, and
                               // the one on the constructor list will go out of
                               // scope after this function
                               this->ptxCode.c_str());
  }

} // ::owl
