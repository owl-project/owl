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

#include "Group.h"

namespace owl {

  /*! a OWL Group / BVH over instances (i.e., a IAS) */
  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public Group::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;
      
      /*! constructor */
      DeviceData(const DeviceContext::SP &device);
      
      DeviceMemory optixInstanceBuffer;

      /*! if we use motion blur, this is used to store all the motoin transforms */
      DeviceMemory motionTransformsBuffer;
      DeviceMemory outputBuffer;

      /*! cuda function handle for the (automatically generated) kernel
        that runs the instance generation program on the device */
      CUfunction instanceFuncKernel = 0;

      /*! cuda function handle for the (automatically generated) kernel
        that runs the motion instance generation program on the device */
      CUfunction motionInstanceFuncKernel = 0;
    };
    
    /*! construct with given array of groups - transforms can be specified later */
    InstanceGroup(Context *const context,
                  size_t numChildren,
                  Group::SP *groups,
                  unsigned int buildFlags,
                  bool useInstanceProgram);

    /*! pretty-printer, for printf-debugging */
    std::string toString() const override;
    
    /*! set given child to given group */
    void setChild(size_t childID, Group::SP child);
                  
    /*! set transformation matrix of given child */
    void setTransform(size_t childID, const affine3f &xfm);

    /*! set transformation matrix of given child */
    void setTransforms(uint32_t timeStep,
                       const float *floatsForThisStimeStep,
                       OWLMatrixFormat matrixFormat);

    /* set instance IDs to use for the children - MUST be an array of
       children.size() items */
    void setInstanceIDs(const uint32_t *instanceIDs);

    /* set visibility masks to use for the children - MUST be an array of
       children.size() items */
    void setVisibilityMasks(const uint8_t *visibilityMasks);

    /*! set instance program to run for this IAS */
    void setInstanceProg(Module::SP module,
                         const std::string &progName);
    
    /*! set motion instance program to run for this IAS */
    void setMotionInstanceProg(Module::SP module,
                               const std::string &progName);
    
    /*! build the CUDA instance program kernel (if instance prog is set) */
    void buildInstanceProg();

    /*! build the CUDA motion instance program kernel (if motion instance prog is set) */
    void buildMotionInstanceProg();

    void buildAccel(LaunchParams::SP launchParams = nullptr) override;
    void refitAccel(LaunchParams::SP launchParams = nullptr) override;

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;

    template<bool FULL_REBUILD>
    void staticBuildOn(const DeviceContext::SP &device);
    template<bool FULL_REBUILD>
    void motionBlurBuildOn(const DeviceContext::SP &device);

    template<bool FULL_REBUILD>
    void staticDeviceBuildOn(const DeviceContext::SP &device, LaunchParams::SP launchParams);
    template<bool FULL_REBUILD>
    void motionBlurDeviceBuildOn(const DeviceContext::SP &device, LaunchParams::SP launchParams);
    
    /*! return the SBT offset to use for this group - SBT offsets for
      instnace groups are always 0 */
    int getSBTOffset() const override { return 0; }
    
    /*! number of children in this group */
    size_t numChildren;

    /*! the list of children - note we do have to keep them both in
      the ll layer _and_ here for the refcounting to work; the
      transforms are only stored once, on the ll layer */
    std::vector<Group::SP>  children;
    
    /*! set of transform matrices for t=0 and t=1, respectively. if we
      don't use motion blur, the second one may be unused */
    std::vector<affine3f>   transforms[2];

    /*! vector of instnace IDs to use for these instances - if not
      specified we/optix will fill in automatically using
      instanceID=childID */
    std::vector<uint32_t>   instanceIDs;

    /*! vector of visibility masks to use for these instances - if not
      specified we/optix will fill in automatically using
      visibility=255 */
    std::vector<uint8_t> visibilityMasks;

    constexpr static unsigned int defaultBuildFlags = 
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;

    protected:
    const unsigned int buildFlags;

    /*! if true, we use the instance program to generate the instances */
    bool useInstanceProgram;

    /*! the instance prog to run for this type */
    ProgramDesc instanceProg;

    /*! the motion instance prog to run for this type (if motion blur is enabled) */
    ProgramDesc motionInstanceProg;
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline InstanceGroup::DeviceData &InstanceGroup::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<DeviceData>();
  }
  
} // ::owl
