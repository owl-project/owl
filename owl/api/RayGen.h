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

#include "SBTObject.h"
#include "Module.h"
#include "LaunchParams.h"

namespace owl {

  struct RayGenType : public SBTObjectType {
    typedef std::shared_ptr<RayGenType> SP;
    RayGenType(Context *const context,
               Module::SP module,
               const std::string &progName,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    virtual std::string toString() const { return "RayGenType"; }
    
    Module::SP        module;
    const std::string progName;
  };
  
  struct RayGen : public SBTObject<RayGenType> {
    typedef std::shared_ptr<RayGen> SP;
    
    RayGen(Context *const context,
           RayGenType::SP type);
    
    virtual std::string toString() const { return "RayGen"; }

    void launch(const vec2i &dims);
    void launch(const vec2i &dims, const LaunchParams::SP &launchParams);
  };

} // ::owl

