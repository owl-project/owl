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

#include "Group.h"
#include "Context.h"

namespace owl {

  void Group::buildAccel()
  {
    // lloGroupAccelBuild(context->llo,this->ID);
    context->llo->groupAccelBuild(this->ID);
  }
  
  OptixTraversableHandle Group::getTraversable(int deviceID)
  {
    // return lloGroupGetTraversable(context->llo,this->ID,deviceID);
    return context->llo->groupGetTraversable(this->ID,deviceID);
  }
  
  void UserGeomGroup::buildAccel()
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
    context->llo->groupAccelBuild(this->ID);
    // lloGroupAccelBuild(context->llo,this->ID);
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
  
  InstanceGroup::InstanceGroup(Context *const context,
                               size_t numChildren,
                               Group::SP      *groups,
                               const uint32_t *instIDs,
                               const float    *xfms,
                               OWLMatrixFormat matrixFormat)
    : Group(context,context->groups),
      children(numChildren)
  {
    std::vector<uint32_t> childIDs;
    if (groups) {
      childIDs.resize(numChildren);
      for (int i=0;i<numChildren;i++) {
        assert(groups[i]);
        children[i] = groups[i];
        childIDs[i] = groups[i]->ID;
      }
    }
    
    if (matrixFormat != OWL_MATRIX_FORMAT_OWL)
      throw std::runtime_error("currently only supporting OWL_MATRIX_FORMAT_OWL");
    const affine3f *affineXfms = (const affine3f *)xfms;
    context->llo->instanceGroupCreate(this->ID,
                                      numChildren,
                                      groups?childIDs.data():(uint32_t*)nullptr,
                                      instIDs,
                                      affineXfms);
  }
  
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

    /*! set transformation matrix of given child */
  void InstanceGroup::setTransform(int childID,
                                   const affine3f &xfm)
  {
    assert(childID >= 0);
    assert(childID < children.size());

    context->llo->instanceGroupSetTransform(this->ID,
                                            childID,
                                            xfm);
  }

  void InstanceGroup::setChild(int childID, Group::SP child)
  {
    assert(childID >= 0);
    assert(childID < children.size());
    children[childID] = child;
    context->llo->instanceGroupSetChild(this->ID,
                                        childID,
                                        child->ID);
  }
  
} // ::owl
