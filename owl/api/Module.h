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

namespace owl {
  struct Context;
  
  /*! captures the concept of a module that contains one or more
    programs. */
  struct Module : public RegisteredObject {
    typedef std::shared_ptr<Module> SP;

    Module(Context *context, const std::string &ptxCode);
    // Module(const std::string &ptxCode,
    //        ll::id_t llID)
    //   : ptxCode(ptxCode),
    //     llID(llID)
    // {
    //   std::cout << "#owl: created module ..." << std::endl;
    // }
    
    virtual std::string toString() const { return "Module"; }
    
    const std::string ptxCode;
    // const ll::id_t    llID;
  };
  
} // ::owl
