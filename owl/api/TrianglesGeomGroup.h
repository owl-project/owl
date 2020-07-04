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

#include "api/Group.h"

namespace owl {
  namespace ll {
    struct Device;
  };

  struct TrianglesGeomGroup : public GeomGroup {

    // /*! any device-specific data, such as optix handles, cuda device
    //     pointers, etc */
    // struct DeviceSpecific : public GeomGroup::DeviceSpecific {
    // };
    
    TrianglesGeomGroup(Context *const context,
                       size_t numChildren);
    virtual std::string toString() const { return "TrianglesGeomGroup"; }

    void buildAccel() override;
    void refitAccel() override;

    /*! (re-)compute the Group::bounds[2] information for motion blur
      - ie, our _parent_ node may need this */
    void updateMotionBounds();

    DeviceData &getDD(ll::Device *device)
    { assert(device->ID < deviceData.size()); return *deviceData[device->ID]; }
    
    /*! creates the device-specific data for this group */
    Group::DeviceData::SP createOn(ll::Device *device) override
    { return std::make_shared<DeviceData>(); }
    
    /*! low-level accel structure builder for given device */
    template<bool FULL_REBUILD>
    void buildAccelOn(ll::Device *device);
  };

} // ::owl
