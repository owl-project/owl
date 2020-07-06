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

#include "Object.h"

namespace owl {

  struct ObjectRegistry;

  /*! a object that is managed/kept track of in a registry that
      assigns linear IDs (so that, for example, the SBT builder can
      easily iterate over all geometries, all geometry types, etc. The
      sole job of this class is to properly register and unregister
      itself in the given registry when it gets created/destroyed */
  struct RegisteredObject : public ContextObject {

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      virtual ~DeviceData() {}
      
      template<typename T>
      inline T *as() { return dynamic_cast<T*>(this); }
    };

    RegisteredObject(Context *const context,
                     ObjectRegistry &registry);
    ~RegisteredObject();

    /*! creates the device-specific data for this group */
    virtual DeviceData::SP createOn(ll::Device *device)
    { return std::make_shared<DeviceData>(); }

    void createDeviceData(const std::vector<ll::Device *> &devices)
    {
      assert(deviceData.empty());
      for (auto device : devices)
        deviceData.push_back(createOn(device));
    }

    int             ID;
    ObjectRegistry &registry;
    
    std::vector<DeviceData::SP> deviceData;
  };

} // ::owl

