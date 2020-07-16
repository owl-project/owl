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

#include "SBTObject.h"
#include "Module.h"

namespace owl {

  struct LaunchParamsType : public SBTObjectType {
    typedef std::shared_ptr<LaunchParamsType> SP;
    LaunchParamsType(Context *const context,
               size_t varStructSize,
               const std::vector<OWLVarDecl> &varDecls);

    virtual std::string toString() const { return "LaunchParamsType"; }
  };

  /*! an object that stores the variables used for building the launch
      params data - this is all this object does: store values and
      write them when requested */
  struct LaunchParams : public SBTObject<LaunchParamsType> {
    typedef std::shared_ptr<LaunchParams> SP;

    struct DeviceData : public RegisteredObject::DeviceData {
      DeviceData(const DeviceContext::SP &device, size_t  dataSize);
      
      OptixShaderBindingTable sbt = {};

      const size_t         dataSize;
      
      /*! host-size memory for the launch paramters - we have a
          host-side copy, too, so we can leave the launch2D call
          without having to first wait for the cudaMemcpy to
          complete */
      std::vector<uint8_t> hostMemory;
      
      /*! the cuda device memory we copy the launch params to */
      DeviceMemory         deviceMemory;
      
      /*! a cuda stream we can use for the async upload and the
          following async launch */
      cudaStream_t         stream = nullptr;
    };

    
    LaunchParams(Context *const context,
                 LaunchParamsType::SP type);
    
    CUstream getCudaStream(const DeviceContext::SP &device);


    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override
    { return std::make_shared<DeviceData>(device,type->varStructSize); }

    DeviceData &getDD(const DeviceContext::SP &device) const
    {
      assert(device);
      assert(device->ID >= 0 && device->ID < deviceData.size());
      return *deviceData[device->ID]->as<DeviceData>();
    }
      

    
    /*! wait for this launch to complete */
    void sync();
    
    std::string toString() const override { return "LaunchParams"; }
  };

} // ::owl

