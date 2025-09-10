// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "Geom.h"

namespace primer {

  /*! a group is nothing but a list of geoms (that can be
      instantiated).if a logical groups on the primer level contains
      geometries of multiple different types it _may_ require multiple
      different groups and instances on the actual implementation
      level, but that'll be implementation speicific. this version is
      (intentionally) still abstract, so devices have to implement
      this on their own */
  struct Group : Object {
    Group(std::vector<OPGeom> &geoms)
      : geoms(geoms)
    {}

    std::vector<OPGeom> geoms;
  };
  
}
