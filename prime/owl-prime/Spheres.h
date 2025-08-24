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

#include "owl-prime/Context.h"
#include "primer-common/Geom.h"
#include <owl/helper/cuda.h>
#include <owl/owl_host.h>

namespace op {

  using primer::Ray;
  using primer::Hit;

  struct Context;

  /*! a type of geometry that, on the device-end side, will be handled
      by an OWL user geometry type */
  struct UserGeom : public primer::Geom {
    UserGeom(Context *context, uint64_t geomDataValue)
      : primer::Geom(geomDataValue),
        context(context)
    {}
    OWLGeom   geom         = 0;
    Context  *context      = 0;
  };
  
  struct Spheres : public UserGeom {

    struct SBTData {
      uint64_t geomDataValue;
      float4  *spheres;
    };
    
    Spheres(Context *context,
            uint64_t geomDataValue,
            OWLBuffer spheresBuffer, int numSpheres);

    OWLBuffer spheresBuffer = 0;
    int       numSpheres    = 0;

    static OWLVarDecl variables[];
  };
  
} // ::op
