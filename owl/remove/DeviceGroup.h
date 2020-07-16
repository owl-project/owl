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

#pragma once

#include "owl/ll/helper/optix.h"
#include <owl/owl.h>

// namespace owl {
//   namespace ll {

//     // typedef int32_t id_t;

//     // struct DeviceGroup;
//     // typedef DeviceGroup *LLOContext;
    

//     struct Device;
    
//     struct DeviceGroup {
      
//       DeviceGroup(const std::vector<Device *> &devices);
//       ~DeviceGroup();

//       /* create an instance of this object that has properly
//          initialized devices for given cuda device IDs. */
//       static DeviceGroup *create(const int *deviceIDs  = nullptr,
//                                  size_t     numDevices = 0);
      
//       // /*! helper function that enables peer access across all devices */
//       // void enablePeerAccess();

//       const std::vector<Device *> devices;
//     };

//   } // ::owl::ll
// } //::owl
