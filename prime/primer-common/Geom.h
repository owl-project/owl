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

#include "primer-common/Ray.h"
#include "primer-common/Object.h"
#include "primer-common/Context.h"

namespace primer {

  /*! abstraction for a geometry (triangle mesh, set of spheres,
      etc). this is the primer-common version that backends will
      likely override */
  struct Geom : public Object {
    Geom(uint64_t geomDataValue) : geomDataValue(geomDataValue) {}

    /*! pretty-printer (mostly for debugging) */
    std::string toString() const override { return "primer::Geom (abstract)"; }
    
    uint64_t const geomDataValue;
  };

}
