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

#include "InstanceGroup.h"
#include "Context.h"
#include "ll/Device.h"
#include <fstream>

#define LOG(message)                                    \
  if (Context::logging())                               \
    std::cout << "#owl.ll(" << device->ID << "): "      \
              << message                                \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl.ll(" << device->ID << "): "              \
              << message << OWL_TERMINAL_DEFAULT << std::endl

#define CLOG(message)                                   \
  if (Context::logging())                               \
    std::cout << "#owl.ll(" << device->ID << "): "      \
              << message                                \
              << std::endl

#define CLOG_OK(message)                                        \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl.ll(" << device->ID << "): "              \
              << message << OWL_TERMINAL_DEFAULT << std::endl


namespace owl {


  
  InstanceGroup::InstanceGroup(Context *const context,
                               size_t numChildren,
                               Group::SP      *groups)
    : Group(context,context->groups),
      children(numChildren)
  {
    std::vector<uint32_t> childIDs;
    if (groups) {
      childIDs.resize(numChildren);
      for (int i=0;i<numChildren;i++) {
        assert(groups[i]);
        children[i] = groups[i];
        childIDs[i] = groups[i]->ID;
      }
    }

    // context->llo->instanceGroupCreate(this->ID,
    //                                   numChildren,
    //                                   groups?childIDs.data():(uint32_t*)nullptr);

    transforms[0].resize(children.size());
    // context->llo->setTransforms(this->ID,0,transforms[0].data());
  }
  
  
  /*! set transformation matrix of given child */
  void InstanceGroup::setTransform(int childID,
                                   const affine3f &xfm)
  {
    assert(childID >= 0);
    assert(childID < children.size());

    transforms[0][childID] = xfm;
    // context->llo->setTransforms(this->ID,0,transforms[0].data());
  }

  void InstanceGroup::setTransforms(uint32_t timeStep,
                                    const float *floatsForThisStimeStep,
                                    OWLMatrixFormat matrixFormat)
  {
    switch(matrixFormat) {
    case OWL_MATRIX_FORMAT_OWL: {
      transforms[timeStep].resize(children.size());
      memcpy(transforms[timeStep].data(),floatsForThisStimeStep,children.size()*sizeof(affine3f));
    } break;
    default:
      throw std::runtime_error("used matrix format not yet implmeneted for InstanceGroup::setTransforms");
    };
    // context->llo->setTransforms(this->ID,
    //                             timeStep,
    //                             (const affine3f *)transforms[timeStep].data());
  }

  void InstanceGroup::setInstanceIDs(/* must be an array of children.size() items */
                                     const uint32_t *_instanceIDs)
  {
    // if (instanceIDs.empty()) {
    //   for (auto device : context->llo->devices) {
    //     ll::InstanceGroup *ig = (ll::InstanceGroup *)device->checkGetGroup(this->ID);
    //     ig->instanceIDs = instanceIDs.data();
    //   }
    // }
    std::copy(_instanceIDs,_instanceIDs+instanceIDs.size(),instanceIDs.data());
  }
  
  
  void InstanceGroup::setChild(int childID, Group::SP child)
  {
    PING; PRINT(childID); PRINT(child);
    
    assert(childID >= 0);
    assert(childID < children.size());
    children[childID] = child;
    // context->llo->instanceGroupSetChild(this->ID,
    //                                     childID,
    //                                     child->ID);
  }

  void InstanceGroup::buildAccel()
  {
    for (auto device : context->getDevices())
      if (transforms[1].empty())
        staticBuildOn<true>(device);
      else
        motionBlurBuildOn<true>(device);
    
    // throw std::runtime_error("not yet ported");
    // for (auto device : context->getDevices())
    //   device->groupBuildAccel(this->ID);
  }
  
  void InstanceGroup::refitAccel()
  {
    for (auto device : context->getDevices())
      if (transforms[1].empty())
        staticBuildOn<false>(device);
      else
        motionBlurBuildOn<false>(device);
    // for (auto device : context->getDevices())
    //   device->groupRefitAccel(this->ID);
  }





  template<bool FULL_REBUILD>
  void InstanceGroup::staticBuildOn(const DeviceContext::SP &device) 
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;
    
    if (FULL_REBUILD) {
      assert("check does not yet exist" && dd.traversable == 0);
      assert("check does not yet exist" && dd.bvhMemory.empty());
    } else {
      assert("check does not yet exist" && dd.traversable != 0);
      assert("check does not yet exist" && !dd.bvhMemory.empty());
    }
      
    SetActiveGPU forLifeTime(device);
    LOG("building instance accel over "
        << children.size() << " groups");

    // ==================================================================
    // sanity check that that many instances are actualy allowed by optix:
    // ==================================================================
    uint32_t maxInstsPerIAS = 0;
    optixDeviceContextGetProperty
      (optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
       &maxInstsPerIAS,
       sizeof(maxInstsPerIAS));
      
    if (children.size() > maxInstsPerIAS)
      throw std::runtime_error("number of children in instance group exceeds "
                               "OptiX's MAX_INSTANCES_PER_IAS limit");
      
    // ==================================================================
    // create instance build inputs
    // ==================================================================
    OptixBuildInput              instanceInput  {};
    OptixAccelBuildOptions       accelOptions   {};
    //! the N build inputs that go into the builder
    // std::vector<OptixBuildInput> buildInputs(children.size());
    std::vector<OptixInstance>   optixInstances(children.size());

    // for now we use the same flags for all geoms
    // uint32_t instanceGroupInputFlags[1] = { 0 };
    // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

    // now go over all children to set up the buildinputs
    for (int childID=0;childID<children.size();childID++) {
      PRINT(childID);
      PRINT(children[childID]);
      Group::SP child = children[childID];
      assert(child);

      assert(transforms[1].empty());
      const affine3f xfm = transforms[0][childID];

      OptixInstance oi = {};
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
      oi.instanceId        = (instanceIDs.empty())?childID:instanceIDs[childID];
      oi.visibilityMask    = 255;
      oi.sbtOffset         = context->numRayTypes * child->getSBTOffset();
      PRINT(oi.sbtOffset);
      oi.visibilityMask    = 255;
      oi.traversableHandle = child->getTraversable(device);
      assert(oi.traversableHandle);
      
      optixInstances[childID] = oi;
    }

    dd.optixInstanceBuffer.alloc(optixInstances.size()*
                                 sizeof(optixInstances[0]));
    dd.optixInstanceBuffer.upload(optixInstances.data(),"optixinstances");
    
    // ==================================================================
    // set up build input
    // ==================================================================
    instanceInput.type
      = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    instanceInput.instanceArray.instances
      = (CUdeviceptr)dd.optixInstanceBuffer.get();
    instanceInput.instanceArray.numInstances
      = (int)optixInstances.size();
      
    // ==================================================================
    // set up accel uptions
    // ==================================================================
    accelOptions.buildFlags =
      OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
      |
      OPTIX_BUILD_FLAG_ALLOW_UPDATE
      ;
    accelOptions.motionOptions.numKeys = 1;
    if (FULL_REBUILD)
      accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
    else
      accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
      
    // ==================================================================
    // query build buffer sizes, and allocate those buffers
    // ==================================================================
    OptixAccelBufferSizes blasBufferSizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                             &accelOptions,
                                             &instanceInput,
                                             1, // num build inputs
                                             &blasBufferSizes
                                             ));
    
    // ==================================================================
    // trigger the build ....
    // ==================================================================
    const size_t tempSize
      = FULL_REBUILD
      ? blasBufferSizes.tempSizeInBytes
      : blasBufferSizes.tempUpdateSizeInBytes;
    LOG("starting to build/refit "
        << prettyNumber(optixInstances.size()) << " instances, "
        << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
        << prettyNumber(tempSize) << "B in temp data");
      
    DeviceMemory tempBuffer;
    tempBuffer.alloc(tempSize);
      
    if (FULL_REBUILD)
      dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
      
    OPTIX_CHECK(optixAccelBuild(optixContext,
                                /* todo: stream */0,
                                &accelOptions,
                                // array of build inputs:
                                &instanceInput,1,
                                // buffer of temp memory:
                                (CUdeviceptr)tempBuffer.get(),
                                tempBuffer.size(),
                                // where we store initial, uncomp bvh:
                                (CUdeviceptr)dd.bvhMemory.get(),
                                dd.bvhMemory.size(),
                                /* the traversable we're building: */ 
                                &dd.traversable,
                                /* no compaction for instances: */
                                nullptr,0u
                                ));
      
    CUDA_SYNC_CHECK();
    
    // ==================================================================
    // aaaaaand .... clean up
    // ==================================================================
    // TODO: move those free's to the destructor, so we can delay the
    // frees until all objects are done
    tempBuffer.free();
      
    LOG_OK("successfully built instance group accel");
  }
    







  template<bool FULL_REBUILD>
  void InstanceGroup::motionBlurBuildOn(const DeviceContext::SP &device)
  {
    DeviceData &dd = getDD(device);
    auto optixContext = device->optixContext;
    
    if (FULL_REBUILD) {
      assert("check does not yet exist" && dd.traversable == 0);
      assert("check does not yet exist" && dd.bvhMemory.empty());
    } else {
      assert("check does not yet exist" && dd.traversable != 0);
      assert("check does not yet exist" && !dd.bvhMemory.empty());
    }
      
    SetActiveGPU forLifeTime(device);
      LOG("building instance accel over "
          << children.size() << " groups");

      // ==================================================================
      // sanity check that that many instances are actualy allowed by optix:
      // ==================================================================
      uint32_t maxInstsPerIAS = 0;
      optixDeviceContextGetProperty
        (optixContext,
         OPTIX_DEVICE_PROPERTY_LIMIT_MAX_INSTANCES_PER_IAS,
         &maxInstsPerIAS,
         sizeof(maxInstsPerIAS));
      
      if (children.size() > maxInstsPerIAS)
        throw std::runtime_error("number of children in instnace group exceeds "
                                 "OptiX's MAX_INSTANCES_PER_IAS limit");
      
      // ==================================================================
      // build motion transforms
      // ==================================================================
      assert(!transforms[1].empty());
      std::vector<OptixMatrixMotionTransform> motionTransforms(children.size());
      std::vector<box3f> motionAABBs(children.size());
      for (int childID=0;childID<children.size();childID++) {
        Group::SP child = children[childID];
        assert(child);
        OptixMatrixMotionTransform mt = {};
        mt.child                      = child->getTraversable(device);
        mt.motionOptions.numKeys      = 2;
        mt.motionOptions.timeBegin    = 0.f;
        mt.motionOptions.timeEnd      = 1.f;
        mt.motionOptions.flags        = OPTIX_MOTION_FLAG_NONE;

        for (int timeStep = 0; timeStep < 2; timeStep ++ ) {
          const affine3f xfm = transforms[timeStep][childID];
          mt.transform[timeStep][0*4+0]  = xfm.l.vx.x;
          mt.transform[timeStep][0*4+1]  = xfm.l.vy.x;
          mt.transform[timeStep][0*4+2]  = xfm.l.vz.x;
          mt.transform[timeStep][0*4+3]  = xfm.p.x;
          
          mt.transform[timeStep][1*4+0]  = xfm.l.vx.y;
          mt.transform[timeStep][1*4+1]  = xfm.l.vy.y;
          mt.transform[timeStep][1*4+2]  = xfm.l.vz.y;
          mt.transform[timeStep][1*4+3]  = xfm.p.y;
          
          mt.transform[timeStep][2*4+0]  = xfm.l.vx.z;
          mt.transform[timeStep][2*4+1]  = xfm.l.vy.z;
          mt.transform[timeStep][2*4+2]  = xfm.l.vz.z;
          mt.transform[timeStep][2*4+3]  = xfm.p.z;
        }

        PING;
        PRINT(child->bounds[0]);
        PRINT(child->bounds[1]);
        
        motionTransforms[childID] = mt;
        std::cout <<" THIS IS WRONG:" << std::endl;

#if 1
        motionAABBs[childID]
          = xfmBounds(transforms[0][childID],child->bounds[0]);
        motionAABBs[childID].extend(xfmBounds(transforms[1][childID],child->bounds[1]));
        PRINT(motionAABBs[childID]);
#else
        motionAABBs[childID] = box3f(vec3f(-100.f),vec3f(+100.f));
#endif
      }
      // and upload
      dd.motionTransformsBuffer.alloc(motionTransforms.size()*
                                   sizeof(motionTransforms[0]));
      dd.motionTransformsBuffer.upload(motionTransforms.data(),"motionTransforms");
      
      dd.motionAABBsBuffer.alloc(motionAABBs.size()*sizeof(box3f));
      dd.motionAABBsBuffer.upload(motionAABBs.data(),"motionaabbs");
      
      // ==================================================================
      // create instance build inputs
      // ==================================================================
      OptixBuildInput              instanceInput  {};
      OptixAccelBuildOptions       accelOptions   {};
      //! the N build inputs that go into the builder
      // std::vector<OptixBuildInput> buildInputs(children.size());
      std::vector<OptixInstance>   optixInstances(children.size());

     // for now we use the same flags for all geoms
      // uint32_t instanceGroupInputFlags[1] = { 0 };
      // { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT

      // now go over all children to set up the buildinputs
      for (int childID=0;childID<children.size();childID++) {
        Group::SP child = children[childID];
        assert(child);

        OptixTraversableHandle childMotionHandle = 0;
        OPTIX_CHECK(optixConvertPointerToTraversableHandle
                    (optixContext,
                     (CUdeviceptr)(((const uint8_t*)dd.motionTransformsBuffer.get())
                                   +childID*sizeof(motionTransforms[0])
                                   ),
                     OPTIX_TRAVERSABLE_TYPE_MATRIX_MOTION_TRANSFORM,
                     &childMotionHandle));
        
        // OptixInstance &oi    = optixInstances[childID];
        OptixInstance oi    = {};
        oi.transform[0*4+0]  = 1.f;//xfm.l.vx.x;
        oi.transform[0*4+1]  = 0.f;//xfm.l.vy.x;
        oi.transform[0*4+2]  = 0.f;//xfm.l.vz.x;
        oi.transform[0*4+3]  = 0.f;//xfm.p.x;
        
        oi.transform[1*4+0]  = 0.f;//xfm.l.vx.y;
        oi.transform[1*4+1]  = 1.f;//xfm.l.vy.y;
        oi.transform[1*4+2]  = 0.f;//xfm.l.vz.y;
        oi.transform[1*4+3]  = 0.f;//xfm.p.y;
        
        oi.transform[2*4+0]  = 0.f;//xfm.l.vx.z;
        oi.transform[2*4+1]  = 0.f;//xfm.l.vy.z;
        oi.transform[2*4+2]  = 1.f;//xfm.l.vz.z;
        oi.transform[2*4+3]  = 0.f;//xfm.p.z;
        
        oi.flags             = OPTIX_INSTANCE_FLAG_NONE;
        oi.instanceId        = (instanceIDs.empty())?childID:instanceIDs[childID];
        oi.sbtOffset         = context->numRayTypes * child->getSBTOffset();
        oi.visibilityMask    = 1; //255;
        oi.traversableHandle = childMotionHandle; //child->traversable;
        optixInstances[childID] = oi;
      }

      dd.optixInstanceBuffer.alloc(optixInstances.size()*
                                sizeof(optixInstances[0]));
      dd.optixInstanceBuffer.upload(optixInstances.data(),"optixinstances");

      // ==================================================================
      // set up build input
      // ==================================================================
      instanceInput.type
        = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
      
      instanceInput.instanceArray.instances
        = dd.optixInstanceBuffer.d_pointer;
      instanceInput.instanceArray.numInstances
        = (int)optixInstances.size();

      instanceInput.instanceArray.aabbs
        = dd.motionAABBsBuffer.d_pointer;
      instanceInput.instanceArray.numAabbs
        = (int)motionAABBs.size();
      
      // ==================================================================
      // set up accel uption
      // ==================================================================
      accelOptions = {};
      accelOptions.buildFlags =
        OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
        |
        OPTIX_BUILD_FLAG_ALLOW_UPDATE
        ;
      // accelOptions.motionOptions.numKeys = 0;//1;
      if (FULL_REBUILD)
        accelOptions.operation            = OPTIX_BUILD_OPERATION_BUILD;
      else {
        throw std::runtime_error("no implemented");
        accelOptions.operation            = OPTIX_BUILD_OPERATION_UPDATE;
      }
      
      // ==================================================================
      // query build buffer sizes, and allocate those buffers
      // ==================================================================
      OptixAccelBufferSizes blasBufferSizes;
      OPTIX_CHECK(optixAccelComputeMemoryUsage(optixContext,
                                               &accelOptions,
                                               &instanceInput,
                                               1, // num build inputs
                                               &blasBufferSizes
                                               ));
    
      // ==================================================================
      // trigger the build ....
      // ==================================================================
      const size_t tempSize
        = FULL_REBUILD
        ? blasBufferSizes.tempSizeInBytes
        : blasBufferSizes.tempUpdateSizeInBytes;
      LOG("starting to build/refit "
          << prettyNumber(optixInstances.size()) << " instances, "
          << prettyNumber(blasBufferSizes.outputSizeInBytes) << "B in output and "
          << prettyNumber(tempSize) << "B in temp data");
      
      DeviceMemory tempBuffer;
      tempBuffer.alloc(tempSize);
      
      if (FULL_REBUILD)
        dd.bvhMemory.alloc(blasBufferSizes.outputSizeInBytes);
      
      OPTIX_CHECK(optixAccelBuild(optixContext,
                                  /* todo: stream */0,
                                  &accelOptions,
                                  // array of build inputs:
                                  &instanceInput,1,
                                  // buffer of temp memory:
                                  (CUdeviceptr)tempBuffer.get(),
                                  tempBuffer.size(),
                                  // where we store initial, uncomp bvh:
                                  (CUdeviceptr)dd.bvhMemory.get(),
                                  dd.bvhMemory.size(),
                                  /* the traversable we're building: */ 
                                  &dd.traversable,
                                  /* no compaction for instances: */
                                  nullptr,0u
                                  ));

      CUDA_SYNC_CHECK();
    
      // ==================================================================
      // aaaaaand .... clean up
      // ==================================================================
      // TODO: move those free's to the destructor, so we can delay the
      // frees until all objects are done
      tempBuffer.free();
      
      LOG_OK("successfully built instance group accel");
    }
  
}
