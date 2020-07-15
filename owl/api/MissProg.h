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

#include "SBTObject.h"
#include "Module.h"

namespace owl {

  struct MissProgType : public SBTObjectType {
    typedef std::shared_ptr<MissProgType> SP;
    MissProgType(Context *const context,
               Module::SP module,
               const std::string &progName,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    /*! for miss progs there's exactly one programgroup pre object */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      OptixProgramGroup pg;
    };

    DeviceData &getDD(int deviceID) const
    {
      assert(deviceID < deviceData.size());
      return *deviceData[deviceID]->as<DeviceData>();
    }
    DeviceData &getDD(const ll::Device *device) const { return getDD(device->ID); }
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(ll::Device *device) override
    { return std::make_shared<DeviceData>(); }
    
    virtual std::string toString() const { return "MissProgType"; }
    
    Module::SP module;
    const std::string progName;
  };
  
  struct MissProg : public SBTObject<MissProgType> {
    typedef std::shared_ptr<MissProg> SP;
    
    MissProg(Context *const context,
           MissProgType::SP type);
    
    void writeSBTRecord(uint8_t *const sbtRecord,
                        Device *device);
    
    virtual std::string toString() const { return "MissProg"; }
  };

} // ::owl

