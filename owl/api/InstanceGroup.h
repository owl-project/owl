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

#pragma once

#include "Group.h"

namespace owl {

  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;
    
    InstanceGroup(Context *const context,
                  size_t numChildren,
                  Group::SP      *groups);

    void setChild(int childID, Group::SP child);
                  
    /*! set transformation matrix of given child */
    void setTransform(int childID, const affine3f &xfm);

    /*! set transformation matrix of given child */
    void setTransforms(uint32_t timeStep,
                       const float *floatsForThisStimeStep,
                       OWLMatrixFormat matrixFormat);

    void setInstanceIDs(/* must be an array of children.size() items */
                        const uint32_t *instanceIDs);
      
    void buildAccel() override;
    void refitAccel() override;

    /*! creates the device-specific data for this group */
    DeviceData::SP createOn(ll::Device *device) override;
    

    virtual std::string toString() const { return "InstanceGroup"; }

    /*! the list of children - note we do have to keep them both in
        the ll layer _and_ here for the refcounting to work; the
        transforms are only stored once, on the ll layer */
    std::vector<Group::SP>  children;
    /*! set of transform matrices for t=0 and t=1, respectively */
    std::vector<affine3f>   transforms[2];
    std::vector<uint32_t>   instanceIDs;
  };

} // ::owl
