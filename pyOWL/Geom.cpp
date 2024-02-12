// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include "pyOWL/Geom.h"
#include "pyOWL/Context.h"

namespace owl {
  size_t sizeOf(OWLDataType type);
};

namespace pyOWL {

  Geom::Geom(Context *ctx, GeomType::SP type)
    : handle(owlGeomCreate(ctx->handle,type->handle)),
      type(type)
  {
    assert(ctx);
    assert(ctx->handle);
    assert(type);
    assert(type->handle);
    assert(handle);
  }
    
  Geom::SP Geom::create(Context *ctx, GeomType::SP type)
  {
    assert(ctx);
    assert(ctx->handle);
    assert(type);
    assert(type->handle);
    return std::make_shared<Geom>(ctx,type);
  }

  void Geom::setVertices(Buffer::SP vertices)
  {
    assert(this);
    assert(handle);
    assert(vertices->handle);
    assert(vertices->count);
    owlTrianglesSetVertices(handle,
                            vertices->handle,
                            vertices->count,
                            owl::sizeOf(vertices->type),
                            0);
  }
  
  void Geom::setIndices(Buffer::SP indices)
  {
    assert(this);
    assert(handle);
    assert(indices->handle);
    assert(indices->count);
    owlTrianglesSetIndices(handle,
                           indices->handle,
                           indices->count,
                           owl::sizeOf(indices->type),
                           0);
  }
    
  void Geom::setPrimCount(int primCount)
  {
    assert(this);
    assert(handle);
    owlGeomSetPrimCount(handle,primCount);
  }
    
}
