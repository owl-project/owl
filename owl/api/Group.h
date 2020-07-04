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

#include "RegisteredObject.h"
#include "Geometry.h"
#include "ll/DeviceMemory.h"
#include "ll/Device.h"

namespace owl {

  using ll::DeviceMemory;
  
  /*! any sort of group */
  struct Group : public RegisteredObject {
    typedef std::shared_ptr<Group> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      OptixTraversableHandle traversable = 0;
      DeviceMemory           bvhMemory;
    };

    /*! creates the device-specific data for this group */
    virtual DeviceData::SP createOn(ll::Device *device) = 0;
    
    Group(Context *const context,
          ObjectRegistry &registry)
      : RegisteredObject(context,registry)
    {}
    virtual std::string toString() const { return "Group"; }
    virtual void buildAccel() = 0;
    virtual void refitAccel() = 0;

    OptixTraversableHandle getTraversable(int deviceID)
    { return deviceData[deviceID]->traversable; }
    
    void createDeviceData(const std::vector<ll::Device *> &devices);

    /*! bounding box for t=0 and t=1, respectively; for motion
        blur. */
    box3f bounds[2];
    std::vector<DeviceData::SP> deviceData;
  };

  /*! a group containing geometries */
  struct GeomGroup : public Group {
    typedef std::shared_ptr<GeomGroup> SP;

    // /*! any device-specific data, such as optix handles, cuda device
    //     pointers, etc */
    
    
    GeomGroup(Context *const context,
              size_t numChildren);
    void setChild(int childID, Geom::SP child);
    
    virtual std::string toString() const { return "GeomGroup"; }
    std::vector<Geom::SP> geometries;
  };

} // ::owl
