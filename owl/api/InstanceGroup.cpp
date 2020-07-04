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

#include "InstanceGroup.h"
#include "Context.h"
#include "ll/Device.h"

namespace owl {
  
  InstanceGroup::InstanceGroup(Context *const context,
                               size_t numChildren,
                               Group::SP      *groups)
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

    // context->llo->instanceGroupCreate(this->ID,
    //                                   numChildren,
    //                                   groups?childIDs.data():(uint32_t*)nullptr);

    transforms[0].resize(children.size());
    // context->llo->setTransforms(this->ID,0,transforms[0].data());
  }
  
  
  /*! set transformation matrix of given child */
  void InstanceGroup::setTransform(int childID,
                                   const affine3f &xfm)
  {
    assert(childID >= 0);
    assert(childID < children.size());

    transforms[0][childID] = xfm;
    // context->llo->setTransforms(this->ID,0,transforms[0].data());
  }

  void InstanceGroup::setTransforms(uint32_t timeStep,
                                    const float *floatsForThisStimeStep,
                                    OWLMatrixFormat matrixFormat)
  {
    switch(matrixFormat) {
    case OWL_MATRIX_FORMAT_OWL: {
      transforms[timeStep].resize(children.size());
      memcpy(transforms[timeStep].data(),floatsForThisStimeStep,children.size()*sizeof(affine3f));
    } break;
    default:
      throw std::runtime_error("used matrix format not yet implmeneted for InstanceGroup::setTransforms");
    };
    // context->llo->setTransforms(this->ID,
    //                             timeStep,
    //                             (const affine3f *)transforms[timeStep].data());
  }

  void InstanceGroup::setInstanceIDs(/* must be an array of children.size() items */
                                     const uint32_t *_instanceIDs)
  {
    // if (instanceIDs.empty()) {
    //   for (auto device : context->llo->devices) {
    //     ll::InstanceGroup *ig = (ll::InstanceGroup *)device->checkGetGroup(this->ID);
    //     ig->instanceIDs = instanceIDs.data();
    //   }
    // }
    std::copy(_instanceIDs,_instanceIDs+instanceIDs.size(),instanceIDs.data());
  }
  
  
  void InstanceGroup::setChild(int childID, Group::SP child)
  {
    assert(childID >= 0);
    assert(childID < children.size());
    children[childID] = child;
    // context->llo->instanceGroupSetChild(this->ID,
    //                                     childID,
    //                                     child->ID);
  }

  void InstanceGroup::buildAccel()
  {
    throw std::runtime_error("not yet ported");
    // for (auto device : context->llo->devices)
    //   device->groupBuildAccel(this->ID);
  }
  
  void InstanceGroup::refitAccel()
  {
    throw std::runtime_error("not yet ported");
    // for (auto device : context->llo->devices)
    //   device->groupRefitAccel(this->ID);
  }
  
}
