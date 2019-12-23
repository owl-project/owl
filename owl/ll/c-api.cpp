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

/*! \file include/owl/ll.h Implements the dynamically linkable "C-API"
 *  for the low-level owl::ll abstraction layer */

// public API:
#include "../include/owl/ll.h"
// internal C++ classes that implement this API
#include "owl/ll/DeviceGroup.h"

#ifndef NDEBUG
# define EXCEPTIONS_ARE_FATAL 1
#endif

namespace owl {
  namespace ll {

    std::string lastErrorText = "";

    template<typename Lambda>
    inline LLOResult squashExceptions(const Lambda &fun)
    {
      try {
        fun();
        return LLO_SUCCESS;
      } catch (const std::runtime_error &e) {
#if EXCEPTIONS_ARE_FATAL
        std::cerr << "Fatal error: " << e.what() << std::endl;
        exit(1);
#else
        lastErrorText = e.what();
        return LLO_UNKNOWN_ERROR;
#endif
      }
    }
    


    extern "C" OWL_LL_INTERFACE
    LLOContext lloContextCreate(const int32_t *deviceIDs,
                                int32_t        numDeviceIDs)
    {
      DeviceGroup *dg = DeviceGroup::create(deviceIDs,numDeviceIDs);
      return (LLOContext)dg;
    }

      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloContextDestroy(LLOContext llo)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          DeviceGroup::destroy(dg);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloLaunch2D(LLOContext llo,
                          int32_t rayGenID,
                          int32_t launchDimX,
                          int32_t launchDimY)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->launch(rayGenID,vec2i(launchDimX,launchDimY));
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocModules(LLOContext llo,
                              int numModules)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocModules(numModules);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocGroups(LLOContext llo,
                             int numGroups)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocGroups(numGroups);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocGeoms(LLOContext llo,
                            int numGeoms)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocGeoms(numGeoms);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocGeomTypes(LLOContext llo,
                                int numGeomTypes)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocGeomTypes(numGeomTypes);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocMissProgs(LLOContext llo,
                                int numMissProgs)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocMissProgs(numMissProgs);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocRayGens(LLOContext llo,
                              int32_t    rayGenProgCount)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocRayGens(rayGenProgCount);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloGeomTypeCreate(LLOContext llo,
                                int32_t    geomTypeID,
                                size_t     sizeOfSBTData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->geomTypeCreate(geomTypeID,
                             sizeOfSBTData);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloGeomTypeIntersect(LLOContext llo,
                                   int32_t geomTypeID,
                                   int32_t rayTypeID,
                                   int32_t moduleID,
                                   const char *programName)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setGeomTypeIntersect(geomTypeID,
                                   rayTypeID,
                                   moduleID,
                                   programName);
        });
    }

    /*! set bounding box program for given geometry type, using a
      bounding box program to be called on the device. note that
      unlike other programs (intersect, closesthit, anyhit) these
      programs are not 'per ray type', but exist only once per
      geometry type. obviously only allowed for user geometry
      typed. */
    extern "C" OWL_LL_INTERFACE
    LLOResult lloGeomTypeBoundsProgDevice(LLOContext llo,
                                          int32_t geomTypeID,
                                          int32_t moduleID,
                                          const char *programName,
                                          size_t geomDataSize)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setGeomTypeBoundsProgDevice(geomTypeID,
                                          moduleID,
                                          programName,
                                          geomDataSize);
        });
    }

    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloTrianglesGeomCreate(LLOContext llo,
                                     /*! ID of the geometry to create */
                                     int32_t    geomID,
                                     /*! ID of the geometry *type* to
                                       use for this geometry (this is
                                       what defines the SBT data size,
                                       closest hit program, etc */
                                     int32_t    geomTypeID)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->trianglesGeomCreate(geomID,geomTypeID);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloUserGeomCreate(LLOContext llo,
                                /*! ID of the geometry to create */
                                int32_t    geomID,
                                /*! ID of the geometry *type* to
                                  use for this geometry (this is
                                  what defines the SBT data size,
                                  closest hit program, etc */
                                int32_t    geomTypeID,
                                int32_t    numPrims)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->userGeomCreate(geomID,
                             geomTypeID,
                             numPrims);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloGeomTypeClosestHit(LLOContext llo,
                                    int32_t geomTypeID,
                                    int32_t rayTypeID,
                                    int32_t moduleID,
                                    const char *programName)
      
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setGeomTypeClosestHit(geomTypeID,
                                    rayTypeID,
                                    moduleID,
                                    programName);
        });
    }


    extern "C" OWL_LL_INTERFACE
    LLOResult lloTrianglesGeomSetVertexBuffer(LLOContext llo,
                                              int32_t    geomID,
                                              int32_t    bufferID,
                                              int32_t    count,
                                              int32_t    stride,
                                              int32_t    offset)
      
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->trianglesGeomSetVertexBuffer(geomID,
                                           bufferID,
                                           count,
                                           stride,
                                           offset);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloTrianglesGeomSetIndexBuffer(LLOContext llo,
                                             int32_t    geomID,
                                             int32_t    bufferID,
                                             int32_t    count,
                                             int32_t    stride,
                                             int32_t    offset)
      
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->trianglesGeomSetIndexBuffer(geomID,
                                          bufferID,
                                          count,
                                          stride,
                                          offset);
        });
    }



    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloTrianglesGeomGroupCreate(LLOContext llo,
                                          int32_t        groupID,
                                          const int32_t *geomIDs,
                                          int32_t        numGeomIDs)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->trianglesGeomGroupCreate(groupID,geomIDs,numGeomIDs);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloUserGeomGroupCreate(LLOContext llo,
                                          int32_t        groupID,
                                          const int32_t *geomIDs,
                                          int32_t        numGeomIDs)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->userGeomGroupCreate(groupID,geomIDs,numGeomIDs);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloInstanceGroupCreate(LLOContext llo,
                                          int32_t        groupID,
                                          const int32_t *childGroupIDs,
                                          int32_t        numChildGroupIDs)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->instanceGroupCreate(groupID,childGroupIDs,numChildGroupIDs);
        });
    }

    /*! set a buffer of bounding boxes that this user geometry will
      use when building the accel structure. this is one of multiple
      ways of specifying the bounding boxes for a user gometry (the
      other two being a) setting the geometry type's boundsFunc, or b)
      setting a host-callback fr computing the bounds). Only one of
      the three methods can be set at any given time */
    extern "C" OWL_LL_INTERFACE
    LLOResult lloUserGeomSetBoundsBuffer(LLOContext llo,
                                         int32_t geomID,
                                         int32_t bufferID)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->userGeomSetBoundsBuffer(geomID,
                                      bufferID);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloModuleCreate(LLOContext llo,
                              int32_t moduleID,
                              const char *ptxCode)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->moduleCreate(moduleID,ptxCode);
        });
    }

    /*! (re-)builds the modules that have been set via
     *  lloModuleCreate */
    extern "C" OWL_LL_INTERFACE
    LLOResult lloBuildModules(LLOContext llo)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->buildModules();
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloRayGenCreate(LLOContext  llo,
                              /*! ID of ray gen prog to create */
                              int32_t     programID,
                              /*! ID of module in which to look for that program */
                              int32_t     moduleID,
                              /*! name of the program */
                              const char *programName,
                              /*! size of that program's SBT data */
                              size_t      dataSizeOfRayGen)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setRayGen(programID,moduleID,programName,dataSizeOfRayGen);
        });
    }
  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloMissProgCreate(LLOContext  llo,
                                /*! ID of ray gen prog to create */
                                int32_t     programID,
                                /*! ID of module in which to look for that program */
                                int32_t     moduleID,
                                /*! name of the program */
                                const char *programName,
                                /*! size of that program's SBT data */
                                size_t      dataSizeOfMissProg)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setMissProg(programID,moduleID,programName,dataSizeOfMissProg);
        });
    }
  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloBuildPrograms(LLOContext llo)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->buildPrograms();
        });
    }
  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloCreatePipeline(LLOContext llo)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->createPipeline();
        });
    }
      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloHostPinnedBufferCreate(LLOContext llo,
                                        /*! ID of buffer to create */
                                        int32_t bufferID,
                                        /*! number of elements */
                                        size_t sizeInBytes)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->hostPinnedBufferCreate(bufferID,sizeInBytes,1);
        });
    }
      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloDeviceBufferCreate(LLOContext llo,
                                    /*! ID of buffer to create */
                                    int32_t bufferID,
                                    /*! number of elements */
                                    size_t sizeInBytes,
                                    const void *initData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->deviceBufferCreate(bufferID,sizeInBytes,1,initData);
        });
    }
      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocBuffers(LLOContext llo,
                              /*! number of buffers valid after this
                               *  function call */
                              int32_t numBuffers)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->allocBuffers(numBuffers);
        });
    }

    /*! builds the SBT's ray gen program entries, using the given
     *  callback to query the app as as to what values to write for a
     *  given ray gen program */
    extern "C" OWL_LL_INTERFACE
    LLOResult lloSbtRayGensBuild(LLOContext llo,
                                 LLOWriteRayGenDataCB writeRayGenDataCB,
                                 const void *callbackData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->sbtRayGensBuild(owl::ll::WriteRayGenDataCB(writeRayGenDataCB),
                              callbackData);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloSbtHitProgsBuild(LLOContext llo,
                                  LLOWriteHitProgDataCB writeHitProgDataCB,
                                  const void *callbackData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->sbtHitProgsBuild(owl::ll::WriteHitProgDataCB(writeHitProgDataCB),
                               callbackData);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloSbtMissProgsBuild(LLOContext llo,
                                   LLOWriteMissProgDataCB writeMissProgDataCB,
                                   const void *callbackData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->sbtMissProgsBuild(owl::ll::WriteMissProgDataCB(writeMissProgDataCB),
                                callbackData);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloGroupBuildPrimitiveBounds(LLOContext llo,
                                           int32_t    groupID,
                                           size_t     maxGeomDataSize,
                                           LLOWriteUserGeomBoundsDataCB cb,
                                           const void *cbData)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->groupBuildPrimitiveBounds
            (groupID,maxGeomDataSize,
             owl::ll::WriteUserGeomBoundsDataCB(cb),
             cbData);
        });
    }


    
    extern "C" OWL_LL_INTERFACE
    size_t lloGetDeviceCount(LLOContext llo)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->getDeviceCount();
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return size_t(-1);
      }
    }
  
    /*! returns the device-side pointer of the given buffer, on the
     *  given device */
    extern "C" OWL_LL_INTERFACE
    const void *lloBufferGetPointer(LLOContext llo,
                                    int32_t    bufferID,
                                    int32_t    deviceID)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->bufferGetPointer(bufferID,deviceID);
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return nullptr;
      }
    }
       
    /*! returns the device-side pointer of the given buffer, on the
     *  given device */
    extern "C" OWL_LL_INTERFACE
    OptixTraversableHandle lloGroupGetTraversable(LLOContext llo,
                                                  int32_t    groupID,
                                                  int32_t    deviceID)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->groupGetTraversable(groupID,deviceID);
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return (OptixTraversableHandle)0;
      }
    }

    extern "C" OWL_LL_INTERFACE
    uint32_t lloGroupGetSbtOffset(LLOContext llo,
                                  int32_t    groupID)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->groupGetSBTOffset(groupID);
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return (OptixTraversableHandle)0;
      }
    }
    

  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloGroupAccelBuild(LLOContext llo,
                                 int32_t    groupID)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->groupBuildAccel(groupID);
        });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloInstanceGroupSetTransform(LLOContext llo,
                                           int32_t    groupID,
                                           int32_t    childID,
                                           const float *xfm)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->instanceGroupSetTransform(groupID,childID,
                                        *(const affine3f*)xfm);
        });
    }
        
    extern "C" OWL_LL_INTERFACE
    LLOResult lloInstanceGroupSetChild(LLOContext llo,
                                       int32_t    groupID,
                                       int32_t    childID,
                                       int32_t    childGroupID,
                                       const float *xfm)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->instanceGroupSetChild(groupID,childID,childGroupID,
                                    *(const affine3f*)xfm);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloGeomGroupSetChild(LLOContext llo,
                                   int32_t    groupID,
                                   int32_t    childNo,
                                   int32_t    childID)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->geomGroupSetChild(groupID,childNo,childID);
        });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloSetMaxInstancingDepth(LLOContext llo,
                                       int32_t maxInstanceDepth)
    {
      return squashExceptions
        ([&](){
          DeviceGroup *dg = (DeviceGroup *)llo;
          dg->setMaxInstancingDepth(maxInstanceDepth);
        });
    }
  

  } // ::owl::ll
} //::owl
