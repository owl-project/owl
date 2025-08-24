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

#include "owl-prime/Triangles.h"
#include "owl-prime/Context.h"

namespace op {

  OWLVarDecl Triangles::variables[]
  = {
     { "geomDataValue", OWL_LONG, OWL_OFFSETOF(Triangles::SBTData,geomDataValue) },
     { nullptr }
  };
  
  Triangles::Triangles(Context *context,
                       /*! user provided data value to be stored
                               in the hit */
                       uint64_t geomDataValue,
                       OWLBuffer vertexBuffer,
                       int numVertices,
                       OWLBuffer indexBuffer,
                       int numIndices)
    : primer::Geom(geomDataValue),
      context(context),
      vertexBuffer(vertexBuffer),
      numVertices(numVertices),
      indexBuffer(indexBuffer),
      numIndices(numIndices)
  {
    geom = owlGeomCreate(context->owl,
                         context->meshGeomType);
    owlTrianglesSetVertices(geom,vertexBuffer,
                            numVertices,sizeof(vec3f),0);
    owlTrianglesSetIndices(geom,indexBuffer,
                           numIndices,sizeof(vec3i),0);
    owlGeomSet1l(geom,"geomDataValue",geomDataValue);
  }

} // ::op
