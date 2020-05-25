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

#include "Device.h"
#include "owl/common/parallel/parallel_for.h"

#define LOG(message)      \
  if (Context::logging()) \
  std::cout << "#owl.ll(" << context->owlDeviceID << "): "      \
  << message                                                    \
  << std::endl

#define LOG_OK(message)                                 \
  if (Context::logging()) \
  std::cout << OWL_TERMINAL_GREEN                       \
  << "#owl.ll(" << context->owlDeviceID << "): "        \
  << message << OWL_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  if (Context::logging()) \
  std::cout << "#owl.ll(" << owlDeviceID << "): "       \
  << message                                            \
  << std::endl

#define CLOG_OK(message)                                \
  if (Context::logging()) \
  std::cout << OWL_TERMINAL_GREEN                       \
  << "#owl.ll(" << owlDeviceID << "): "                 \
  << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    /*! set given child's instance transform. groupID must be a
      valid instance group, childID must be wihtin
      [0..numChildren) */
    void Device::instanceGroupSetTransform(int groupID,
                                           int childNo,
                                           const affine3f &xfm)
    {
      InstanceGroup *ig = checkGetInstanceGroup(groupID);
      assert("check valid child slot" && childNo >= 0);
      assert("check valid child slot" && childNo <  ig->children.size());
      
      if (ig->transforms.empty())
        ig->transforms.resize(ig->children.size());
      ig->transforms[childNo] = xfm;
    }
    
    /*! set given child to {childGroupID+xfm}  */
    void Device::instanceGroupSetChild(int groupID,
                                       int childNo,
                                       int childGroupID)
    {
      InstanceGroup *ig = checkGetInstanceGroup(groupID);
      Group *newChild = checkGetGroup(childGroupID);
      if (ig->transforms.empty())
        ig->transforms.resize(ig->children.size());
      Group *oldChild = ig->children[childNo];
      if (oldChild)
        oldChild->numTimesReferenced--;
      ig->children[childNo] = newChild;
      newChild->numTimesReferenced++;
    }

    void Device::instanceGroupCreate(/*! the group we are defining */
                                     int groupID,
                                     size_t childCount,
                                     /* list of children. list can be
                                        omitted by passing a nullptr, but if
                                        not null this must be a list of
                                        'childCount' valid group ID */
                                     const uint32_t *childGroupIDs,
                                     const uint32_t *instIDs,
                                     const affine3f *xfms)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] == nullptr);
        
      InstanceGroup *group
        = new InstanceGroup(childCount);
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;

      if (instIDs)
        group->instanceIDs.resize(childCount);
      if (xfms)
        group->transforms.resize(childCount);
      owl::parallel_for
        (childCount,[&](size_t childID){
          if (childGroupIDs) {
            int childGroupID = childGroupIDs[childID];
            assert("check geom child child group ID is valid"
                   && childGroupID >= 0);
            assert("check geom child child group ID is valid"
                   && childGroupID <  groups.size());
            Group *childGroup = groups[childGroupID];
            assert("check referened child groups is valid" && childGroup != nullptr);
            childGroup->numTimesReferenced++;
            group->children[childID] = childGroup;
          }
          if (xfms)
            group->transforms[childID] = xfms[childID];
          if (instIDs)
            group->instanceIDs[childID] = instIDs[childID];
        },8*1024);
    }

    void InstanceGroup::destroyAccel(Context *context) 
    {
      context->pushActive();
      if (traversable) {
        bvhMemory.free();
        traversable = 0;
      }
      context->popActive();
    }
    
    void InstanceGroup::buildAccel(Context *context) 
    {
      assert("check does not yet exist" && traversable == 0);
      assert("check does not yet exist" && bvhMemory.empty());
      
      context->pushActive();
      LOG("building instance accel over "
          << children.size() << " groups");

      // ==================================================================
      // sanity check that that many instances are actualy allowed by optix:
      // ==================================================================
      uint32_t maxInstsPerIAS = 0;
      optixDeviceContextGetProperty
        (context->optixContext,
         OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
         &maxInstsPerIAS,
         sizeof(maxInstsPerIAS));

      // uint32_t maxInstanceID = 0;
      // optixDeviceContextGetProperty
      //   (context->optixContext,
      //    OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCE_ID,
      //    &maxInstanceID,
      //    sizeof(maxInstanceID));
      // PRINT(maxInstanceID);
      // exit(0);
      

      
      if (children.size() > maxInstsPerIAS)
        throw std::runtime_error("number of children in instnace group exceeds "
                                 "OptiX's MAX_INSTANCES_PER_IAS limit");
      
      // ==================================================================
      // create instance build inputs
      // ==================================================================
      OptixBuildInput              instanceInput  {};
      OptixAccelBuildOptions       accelOptions   {};
      //! the N build inputs that go into the builder
      std::vector<OptixBuildInput> buildInputs(children.size());
      std::vector<OptixInstance>   optixInstances(children.size());

     // for now we use the same flags for all geoms
      uint32_t instanceGroupInputFlags[1] = { 0 };
      // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

      // now go over all children to set up the buildinputs
      for (int childID=0;childID<children.size();childID++) {
        Group *child = children[childID];
        assert(child);
        
        const affine3f xfm
          = transforms.empty()
          ? affine3f(owl::common::one)
          : transforms[childID];

        OptixInstance &oi    = optixInstances[childID];
        oi.transform[0*4+0]  = xfm.l.vx.x;
        oi.transform[0*4+1]  = xfm.l.vy.x;
        oi.transform[0*4+2]  = xfm.l.vz.x;
        oi.transform[0*4+3]  = xfm.p.x;
        
        oi.transform[1*4+0]  = xfm.l.vx.y;
        oi.transform[1*4+1]  = xfm.l.vy.y;
        oi.transform[1*4+2]  = xfm.l.vz.y;
        oi.transform[1*4+3]  = xfm.p.y;
        
        oi.transform[2*4+0]  = xfm.l.vx.z;
        oi.transform[2*4+1]  = xfm.l.vy.z;
        oi.transform[2*4+2]  = xfm.l.vz.z;
        oi.transform[2*4+3]  = xfm.p.z;
        
        oi.flags             = OPTIX_INSTANCE_FLAG_NONE;
        oi.instanceId        = instanceIDs.empty()?childID:instanceIDs[childID];
        // PRINT(oi.instanceId);
        oi.visibilityMask    = 255;
        oi.sbtOffset         = context->numRayTypes * child->getSBTOffset();
        oi.visibilityMask    = 255;
        assert(child->traversable);
        oi.traversableHandle = child->traversable;
      }

      optixInstanceBuffer.alloc(optixInstances.size()*
                                sizeof(optixInstances[0]));
      optixInstanceBuffer.upload(optixInstances.data(),"optixinstances");
    
      // ==================================================================
      // set up build input
      // ==================================================================
      instanceInput.type
        = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      instanceInput.instanceArray.instances
        = (CUdeviceptr)optixInstanceBuffer.get();
      instanceInput.instanceArray.numInstances
        = (int)optixInstances.size();
      
      // ==================================================================
      // set up accel uptions
      // ==================================================================
      accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        ;
      accelOptions.motionOptions.numKeys = 1;
      accelOptions.operation             = OPTIX_BUILD_OPERATION_BUILD;
      
      // ==================================================================
      // query build buffer sizes, and allocate those buffers
      // ==================================================================
      OptixAccelBufferSizes bufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(context->optixContext,
                                               &accelOptions,
                                               &instanceInput,
                                               1, // num build inputs
                                               &bufferSizes
                                               ));
    
      // ==================================================================
      // trigger the build ....
      // ==================================================================
      
      LOG("starting to build "
          << prettyNumber(optixInstances.size()) << " instances, "
          << prettyNumber(bufferSizes.outputSizeInBytes) << "B in output and "
          << prettyNumber(bufferSizes.tempSizeInBytes) << "B in temp data");
      
      DeviceMemory tempBuildBuffer;
      tempBuildBuffer.alloc(bufferSizes.tempSizeInBytes);
      
      DeviceMemory &outputBuffer = bvhMemory;
      outputBuffer.alloc(bufferSizes.outputSizeInBytes);
            
      OPTIX_CHECK(optixAccelBuild(context->optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  &instanceInput,1,
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuildBuffer.get(),
                                  tempBuildBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)outputBuffer.get(),
                                  outputBuffer.size(),
                                  /* the traversable we're building: */ 
                                  &traversable,
                                  /* no compaction for instances: */
                                  nullptr,0u
                                  ));
      
      CUDA_SYNC_CHECK();
    
      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      // TODO: move those free's to the destructor, so we can delay the
      // frees until all objects are done
      tempBuildBuffer.free();
      context->popActive();
      
      LOG_OK("successfully built instance group accel");
    }
    

  } // ::owl::ll
} //::owl
