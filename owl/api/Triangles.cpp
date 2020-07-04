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

#include "api/Triangles.h"
#include "api/Context.h"
#include "ll/Device.h"

namespace owl {
  
  TrianglesGeomType::TrianglesGeomType(Context *const context,
                                       size_t varStructSize,
                                       const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {
    /*! nothing special - all inherited */
  }

  TrianglesGeom::TrianglesGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
  {
    // context->llo->trianglesGeomCreate(this->ID,geometryType->ID);
    // for (auto device : context->llo->devices)
    //   llGeom.push_back((ll::TrianglesGeom*)device->checkGetGeom(this->ID));
  }

  /*! set the vertex array (if vector size is 1), or set/enable
    motion blur via multiple time steps, if vector size >= 0 */
  void TrianglesGeom::setVertices(const std::vector<Buffer::SP> &vertexArrays,
                                  /*! the number of vertices in each time step */
                                  size_t count,
                                  size_t stride,
                                  size_t offset)
  {
    vertex.buffers = vertexArrays;
    vertex.count   = count;
    vertex.stride  = stride;
    vertex.offset  = offset;
    std::vector<int32_t> vertexBufferIDs(vertexArrays.size());
    for (int i=0;i<vertexArrays.size();i++)
      vertexBufferIDs[i] = vertexArrays[i]->ID;
    // for (auto device : context->llo->devices)
    //   device->trianglesGeomSetVertexBuffers(this->ID,
    //                                         vertexBufferIDs,count,stride,offset);
  }
  
  void TrianglesGeom::setIndices(Buffer::SP indices,
                                 size_t count,
                                 size_t stride,
                                 size_t offset)
  {
    index.buffer = indices;
    index.count  = count;
    index.stride = stride;
    index.offset = offset;
    // for (auto device : context->llo->devices)
    //   device->trianglesGeomSetIndexBuffer(this->ID,
    //                                       indices->ID, count, stride, offset);
  }

} //::owl
