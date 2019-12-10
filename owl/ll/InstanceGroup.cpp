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

#define LOG(message)                                            \
  std::cout << "#owl.ll(" << context->owlDeviceID << "): "      \
  << message                                                    \
  << std::endl

#define LOG_OK(message)                                 \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << context->owlDeviceID << "): "        \
  << message << GDT_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  std::cout << "#owl.ll(" << owlDeviceID << "): "       \
  << message                                            \
  << std::endl

#define CLOG_OK(message)                                \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll(" << owlDeviceID << "): "                 \
  << message << GDT_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    void Device::instanceGroupCreate(/*! the group we are defining */
                                     int groupID,
                                     /* list of children. list can be
                                        omitted by passing a nullptr, but if
                                        not null this must be a list of
                                        'childCount' valid group ID */
                                     int *childGroupIDs,
                                     /*! number of children in this group */
                                     int childCount)
    {
      assert("check for valid ID" && groupID >= 0);
      assert("check for valid ID" && groupID < groups.size());
      assert("check group ID is available" && groups[groupID] == nullptr);
        
      assert("check for valid combinations of child list" &&
             ((childGroupIDs == nullptr && childCount == 0) ||
              (childGroupIDs != nullptr && childCount >  0)));
        
      InstanceGroup *group
        = new InstanceGroup(childCount);
      assert("check 'new' was successful" && group != nullptr);
      groups[groupID] = group;
      
      // set children - todo: move to separate (api?) function(s)!?
      assert("currently have to specify all children at creation time" &&
             childCount > 0);
      assert("currently have to specify all children at creation time" &&
             childGroupIDs != nullptr);
      for (int childID=0;childID<childCount;childID++) {
        int childGroupID = childGroupIDs[childID];
        assert("check geom child child group ID is valid" && childGroupID >= 0);
        assert("check geom child child group ID is valid" && childGroupID <  groups.size());
        Group *childGroup = groups[childGroupID];
        assert("check referened child groups is valid" && childGroup != nullptr);
        childGroup->numTimesReferenced++;
        group->children[childID] = childGroup;
      }
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
      assert("check does not yet exist" && !bvhMemory.valid());
      
      context->pushActive();
      LOG("building instance accel over "
          << children.size() << " groups");
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
          ? affine3f(gdt::one)
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
        oi.instanceId        = childID; // ???
        oi.visibilityMask    = 255;
        oi.sbtOffset         = child->getSBTOffset();
        oi.visibilityMask    = 255;
        assert(child->traversable);
        oi.traversableHandle = child->traversable;
      }

      DeviceMemory optixInstanceBuffer;
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
        = optixInstances.size();
      
      // ==================================================================
      // set up accel uptions
      // ==================================================================
      accelOptions.buildFlags
        = OPTIX_BUILD_FLAG_NONE
        | OPTIX_BUILD_FLAG_ALLOW_COMPACTION;
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
      // prepare compaction
      // ==================================================================
    
      DeviceMemory compactedSizeBuffer;
      compactedSizeBuffer.alloc(sizeof(uint64_t));
      
      OptixAccelEmitDesc emitDesc;
      emitDesc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
      emitDesc.result = (CUdeviceptr)compactedSizeBuffer.get();
      // ==================================================================
      // trigger the build ....
      // ==================================================================
      
      LOG("starting to build "
          << prettyNumber(optixInstances.size()) << " instances, "
          << prettyNumber(bufferSizes.outputSizeInBytes) << "B in output and "
          << prettyNumber(bufferSizes.tempSizeInBytes) << "B in temp data");
      
      DeviceMemory tempBuildBuffer;
      tempBuildBuffer.alloc(bufferSizes.tempSizeInBytes);
      
      DeviceMemory outputBuffer;
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
                                  /* we're also querying compacted size: */
                                  &emitDesc,1u
                                  ));
      
      CUDA_SYNC_CHECK();
    
      // ==================================================================
      // perform compaction
      // ==================================================================
      uint64_t compactedSize;
      compactedSizeBuffer.download(&compactedSize);
      
      bvhMemory.alloc(compactedSize);
      OPTIX_CHECK(optixAccelCompact(context->optixContext,
                                    /*TODO: stream:*/0,
                                    // OPTIX_COPY_MODE_COMPACT,
                                    traversable,
                                    (CUdeviceptr)bvhMemory.get(),
                                    bvhMemory.size(),
                                    &traversable));
      CUDA_SYNC_CHECK();
      outputBuffer.free(); // << the UNcompacted, temporary output buffer
      
      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      // TODO: move those free's to the destructor, so we can delay the
      // frees until all objects are done
      tempBuildBuffer.free();
      compactedSizeBuffer.free();
      
      context->popActive();
      
      LOG_OK("successfully built instance group accel");
    }
    

  } // ::owl::ll
} //::owl
