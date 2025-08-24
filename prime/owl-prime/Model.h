// ======================================================================== //
// Copyright 2019-2025 Ingo Wald                                            //
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

#include "primer-common/Model.h"
#include <owl/owl.h>
               
namespace op {
  using namespace owl::common;
  
  using primer::Ray;
  using primer::Hit;
  
  struct Context;
  
  struct Model : public primer::Model {

    Model(Context *context,
          const std::vector<OPGroup>  &groups,
          const std::vector<affine3f> &xfms);

    void trace(Ray *rays,
               Hit *hits,
               int  numRaysAndHits,
               int *activeRayIndices,
               int  numActiveIndices,
               OPTraceFlags flags) override;
    
    void build() override;
    
    Context  *context      = 0;
    OWLGroup  handle       = 0;
  };

} // ::op
