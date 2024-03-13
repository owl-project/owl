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

#include "pyOWL/Module.h"
#include "pyOWL/Buffer.h"
#include "pyOWL/Group.h"

namespace pyOWL {

  struct RayGen : public std::enable_shared_from_this<RayGen>
  {
    typedef std::shared_ptr<RayGen> SP;
    
    RayGen(OWLRayGen handle) : handle(handle) {}
    
    virtual ~RayGen()
    {}

    static RayGen::SP create(Context *ctx,
                             const Module::SP &module,
                             const std::string &typeName,
                             const std::string &funcName);
    
    void launch2D(const std::vector<int> &_fbSize)
    {
      const vec2i fbSize = make_vec2i(_fbSize);
      owlRayGenLaunch2D(handle,fbSize.x,fbSize.y);
    }
    
    void set2i(const std::string &name, const std::vector<int> &vec)
    {
      const vec2i v = make_vec2i(vec);
      owlRayGenSet2i(handle,name.c_str(),v.x,v.y);
    }
    
    void set3f(const std::string &name, const std::vector<float> &vec)
    {
      const vec3f v = make_vec3f(vec);
      owlRayGenSet3f(handle,name.c_str(),v.x,v.y,v.z);
    }

    void setBuffer(const std::string &name, const Buffer::SP &buf)
    { owlRayGenSetBuffer(handle,name.c_str(),buf->handle); }
    
    void setGroup(const std::string &name, const Group::SP &buf)
    { owlRayGenSetGroup(handle,name.c_str(),buf->handle); }
    
    const OWLRayGen handle;
  };
  
}
