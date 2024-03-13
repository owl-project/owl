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

namespace pyOWL {

  struct Group : public std::enable_shared_from_this<Group>
  {
    typedef std::shared_ptr<Group> SP;

    Group(OWLGroup handle) : handle(handle) {}
    
    static Group::SP createTrianglesGG(Context *ctx,
                                       const py::list &list);
    static Group::SP createUserGG(Context *ctx,
                                       const py::list &list);
    static Group::SP createInstanceGroup(Context *ctx,
                                         const py::list &list);
    void buildAccel();
    
    const OWLGroup handle;
  };
  
}

