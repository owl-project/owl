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

#pragma once

#include "pyOWL/GeomType.h"
#include "pyOWL/Buffer.h"
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>

namespace pyOWL {

  struct Geom : public std::enable_shared_from_this<Geom>
  {
    typedef std::shared_ptr<Geom> SP;
    
    Geom(Context *ctx, GeomType::SP type);

    void setPrimCount(int primCount);
    void setVertices(Buffer::SP vertices);
    void setIndices(Buffer::SP indices);

    static Geom::SP create(Context *ctx, GeomType::SP type);

    void setBuffer(const std::string &name, const Buffer::SP &buffer)
    { owlGeomSetBuffer(handle,name.c_str(),buffer->handle); }

    // void set3f(const std::string &name, double x, double y, double z)
    // { owlGeomSet3f(handle,name.c_str(),x,y,z); }
    void set3f(const std::string &name, const std::vector<float> &vec)
    { owlGeomSet3f(handle,name.c_str(),vec[0],vec[1],vec[2]); }
    void set1f(const std::string &name, float &f)
    { owlGeomSet1f(handle,name.c_str(),f); }
    
    std::string toString() { return "Geom"; }
    
    const OWLGeom handle;
    GeomType::SP  type;
  };
  
}
