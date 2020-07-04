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

#include "RegisteredObject.h"
#include "SBTObject.h"
#include "Module.h"
#include "Buffer.h"

#include "ll/Device.h"
#include "ll/DeviceMemory.h"

namespace owl {

  using ll::DeviceMemory;
  
  struct Geom;

  struct ProgramDesc {
    Module::SP  module;
    std::string progName;
  };
  
  struct GeomType : public SBTObjectType {
    typedef std::shared_ptr<GeomType> SP;
    
    GeomType(Context *const context,
             size_t varStructSize,
             const std::vector<OWLVarDecl> &varDecls);
    
    virtual std::string toString() const { return "GeomType"; }
    virtual void setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName);

    std::vector<ProgramDesc> closestHit;
    
    virtual void setAnyHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName);

    std::vector<ProgramDesc> anyHit;
    virtual std::shared_ptr<Geom> createGeom() = 0;
  };

  struct Geom : public SBTObject<GeomType> {
    typedef std::shared_ptr<Geom> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      virtual ~DeviceData() {}
      
      template<typename T>
      inline T *as() { return dynamic_cast<T*>(this); }
    };

    Geom(Context *const context,
         GeomType::SP geomType);
    virtual std::string toString() const { return "Geom"; }

    void createDeviceData(const std::vector<ll::Device *> &devices);

    /*! creates the device-specific data for this group */
    virtual DeviceData::SP createOn(ll::Device *device) = 0;

    std::vector<DeviceData::SP> deviceData;
    
    GeomType::SP geomType;
  };

} // ::owl
