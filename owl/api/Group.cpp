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

#include "Group.h"
#include "Context.h"

namespace owl {

  void Group::buildAccel()
  {
    // lloGroupAccelBuild(context->llo,this->ID);
    context->llo->groupBuildAccel(this->ID);
  }
  
  void Group::refitAccel()
  {
    // lloGroupAccelRefit(context->llo,this->ID);
    context->llo->groupRefitAccel(this->ID);
  }
  
  OptixTraversableHandle Group::getTraversable(int deviceID)
  {
    // return lloGroupGetTraversable(context->llo,this->ID,deviceID);
    return context->llo->groupGetTraversable(this->ID,deviceID);
  }
  
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
    if (FULL_REBUILD)
      context->llo->groupBuildAccel(this->ID);
    else
      context->llo->groupRefitAccel(this->ID);
    // lloGroupAccelBuild(context->llo,this->ID);
  }
  
  void UserGeomGroup::buildAccel()
  {
    buildOrRefit(true);
  }
  void UserGeomGroup::refitAccel()
  {
    buildOrRefit(false);
  }

  

  void GeomGroup::setChild(int childID, Geom::SP child)
  {
    assert(childID >= 0);
    assert(childID < geometries.size());
    geometries[childID] = child;
    context->llo->geomGroupSetChild(this->ID,childID,child->ID);
    // lloGeomGroupSetChild(context->llo,this->ID,childID,child->ID);
  }


    
  GeomGroup::GeomGroup(Context *const context,
                       size_t numChildren)
    : Group(context,context->groups),
      geometries(numChildren)
  {}
  
  TrianglesGeomGroup::TrianglesGeomGroup(Context *const context,
                                 size_t numChildren)
    : GeomGroup(context,numChildren)
  {
    context->llo->trianglesGeomGroupCreate(this->ID,
                                           nullptr,numChildren);
  }
  
  UserGeomGroup::UserGeomGroup(Context *const context,
                                 size_t numChildren)
    : GeomGroup(context,numChildren)
  {
    context->llo->userGeomGroupCreate(this->ID,
                                      nullptr,numChildren);
  }

} // ::owl
