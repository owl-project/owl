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

#pragma once

#include "SBTObject.h"
#include "Module.h"

namespace owl {

  struct LaunchParamsType : public SBTObjectType {
    typedef std::shared_ptr<LaunchParamsType> SP;
    LaunchParamsType(Context *const context,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    virtual std::string toString() const { return "LaunchParamsType"; }
  };

  /*! an object that stores the variables used for building the launch
      params data - this is all this object does: store values and
      write them when requested */
  struct LaunchParams : public SBTObject<LaunchParamsType> {
    typedef std::shared_ptr<LaunchParams> SP;
    
    LaunchParams(Context *const context,
           LaunchParamsType::SP type);

    CUstream getCudaStream(int deviceID);
    
    std::string toString() const override { return "LaunchParams"; }
  };

} // ::owl

