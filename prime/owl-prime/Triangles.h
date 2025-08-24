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

#include "primer-common/Geom.h"
#include <owl/helper/cuda.h>
#include <owl/owl_host.h>

namespace op {

  using primer::Ray;
  using primer::Hit;

  struct Context;
  
  struct Triangles : public primer::Geom {

    struct SBTData {
      uint64_t geomDataValue;
    };
    
    Triangles(Context *context,
              uint64_t geomDataValue,
              OWLBuffer vertexBuffer, int numVertices,
              OWLBuffer indexBuffer, int numIndices);
    
    OWLBuffer vertexBuffer = 0;
    int       numVertices  = 0;
    OWLBuffer indexBuffer  = 0;
    int       numIndices   = 0;
    OWLGeom   geom         = 0;
    Context  *context      = 0;

    static OWLVarDecl variables[];
  };
  
} // ::op
