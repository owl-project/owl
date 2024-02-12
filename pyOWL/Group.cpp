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

#include "pyOWL/Group.h"
#include "pyOWL/Context.h"

namespace pyOWL {

  Group::SP Group::createTrianglesGG(Context *ctx,
                                     const py::list &list)
  {
    std::vector<OWLGeom> geoms;
    for (auto item : list) {
      Geom::SP geom = item.cast<Geom::SP>();
      assert(geom);
      geoms.push_back(geom->handle);
    }
    OWLGroup handle = owlTrianglesGeomGroupCreate(ctx->handle,
                                                  geoms.size(),
                                                  geoms.data());
    assert(handle);
    return std::make_shared<Group>(handle);
  }

  Group::SP Group::createUserGG(Context *ctx,
                                     const py::list &list)
  {
    std::vector<OWLGeom> geoms;
    for (auto item : list) {
      Geom::SP geom = item.cast<Geom::SP>();
      assert(geom);
      geoms.push_back(geom->handle);
    }
    OWLGroup handle = owlUserGeomGroupCreate(ctx->handle,
                                             geoms.size(),
                                             geoms.data());
    assert(handle);
    return std::make_shared<Group>(handle);
  }

  Group::SP Group::createInstanceGroup(Context *ctx,
                                       const py::list &list)
  {
    std::vector<OWLGroup> groups;
    for (auto item : list) {
      Group::SP group = item.cast<Group::SP>();
      assert(group);
      groups.push_back(group->handle);
    }
    OWLGroup handle = owlInstanceGroupCreate(ctx->handle,
                                             groups.size(),
                                             groups.data());
    assert(handle);
    return std::make_shared<Group>(handle);
  }

  void Group::buildAccel()
  {
    assert(handle);
    owlGroupBuildAccel(handle);
  }
  

}
