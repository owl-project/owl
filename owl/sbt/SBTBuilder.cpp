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

#include "owl/owl.h"
#include "optix/common.h"
#include <set>
#include <map>
#include <vector>
#include <stack>
#include <typeinfo>
#include <mutex>

namespace owl {

  using gdt::vec3f;

  struct VarDecl {
    const char *name;
    size_t      offset;
    OWLType     type;
  };
  
  struct SBTObjectType {
    std::vector<VarDecl>       varDecls;
    size_t dataSize;
  };
  template<typename Type>
  struct SBTObject {
    std::vector<VarDecl> varDecls;
    size_t               dataSize;
    typename Type::SP    type;
  };

  struct GeometryType : public SBTObjectType {
    Program::SP intersect;
    Program::SP bounds;
    Program::SP anyHit;
    Program::SP closestHit;
  };
  
  struct Geometry : public SBTObject<GeometryType> {
    GeometryType::SP           type;
    std::vector<unsigned char> data;
  };
  
} // ::owl
