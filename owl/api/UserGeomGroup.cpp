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

#include "api/UserGeomGroup.h"
#include "api/Context.h"
#include "ll/Device.h"

namespace owl {
  
  void UserGeomGroup::buildOrRefit(bool FULL_REBUILD)
  {
    size_t maxVarSize = 0;
    for (auto child : geometries) {
      assert(child);
      assert(child->type);
      maxVarSize = std::max(maxVarSize,child->type->varStructSize);
    }

    // TODO: do this only if there's no explicit bounds buffer set
    context->llo->groupBuildPrimitiveBounds
      (this->ID,maxVarSize,
       [&](uint8_t *output, int devID, int geomID, int childID) {
        assert(childID >= 0 && childID < geometries.size());
        Geom::SP child = geometries[childID];
        assert(child);
        child->writeVariables(output,devID);
      });
    for (auto device : context->llo->devices)
      if (FULL_REBUILD)
        device->groupBuildAccel(this->ID);
      else
        device->groupRefitAccel(this->ID);
  }
  
  void UserGeomGroup::buildAccel()
  {
    buildOrRefit(true);
  }

  void UserGeomGroup::refitAccel()
  {
    buildOrRefit(false);
  }

  UserGeomGroup::UserGeomGroup(Context *const context,
                                 size_t numChildren)
    : GeomGroup(context,numChildren)
  {
    context->llo->userGeomGroupCreate(this->ID,
                                      nullptr,numChildren);
  }

} // ::owl
