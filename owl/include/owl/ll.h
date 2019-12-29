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

/*! \file include/owl/ll.h Defines the dynamically linkable "C-API"
 *  for the low-level owl::ll abstraction layer */

#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif

#include <iostream>
#include <math.h> // using cmath causes issues under Windows
#ifdef _WIN32
#else
#  include <unistd.h>
#endif
#include <stdint.h>

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif

#include <optix.h>
#ifdef _WIN32
#pragma warning( push )
#endif

#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define __PRETTY_FUNCTION__ __FUNCSIG__
#endif

#if defined(_MSC_VER)
#  define OWL_LL_DLL_EXPORT __declspec(dllexport)
#  define OWL_LL_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_LL_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_LL_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_LL_DLL_EXPORT
#  define OWL_LL_DLL_IMPORT
#endif

//#if defined(OWL_LL_DLL_INTERFACE)
//#if _WIN32
#  ifdef llowl_EXPORTS
#    define OWL_LL_INTERFACE OWL_LL_DLL_EXPORT
#  else
#    define OWL_LL_INTERFACE OWL_LL_DLL_IMPORT
#  endif
//#else
//#  define OWL_LL_INTERFACE __attribute__((visibility("default")))
//#endif


#ifdef __cplusplus
extern "C" {
#endif

  typedef struct _LLOContext *LLOContext;

  typedef enum
    {
      /*! no error - api did what it was asked to do */
      LLO_SUCCESS = 0,
      /*! some un-specified error happened. use lloGetLastErrorString
       *  to get a textual description */
      LLO_UNKNOWN_ERROR
    }
    LLOResult;
  
  
  typedef void
  (*LLOWriteUserGeomBoundsDataCB)(uint8_t *userGeomDataToWrite,
                                  int deviceID,
                                  int geomID,
                                  int childID,
                                  const void *cbUserData);
    
  /*! callback with which the app can specify what data is to be
    written into the SBT for a given geometry, ray type, and
    device */
  typedef void
  (*LLOWriteHitProgDataCB)(uint8_t *hitProgDataToWrite,
                           /*! ID of the device we're
                             writing for (differnet
                             devices may need to write
                             different pointers */
                           int deviceID,
                           /*! the geometry ID for which
                             we're generating the SBT
                             entry for */
                           int geomID,
                           /*! the ray type for which
                             we're generating the SBT
                             entry for */
                           int rayType,
                           /*! the raw void pointer the app has passed
                             during sbtHitGroupsBuild() */
                           const void *callBackUserData);
    
  /*! callback with which the app can specify what data is to be
    written into the SBT for a given geometry, ray type, and
    device */
  typedef void
  (*LLOWriteRayGenDataCB)(uint8_t *rayGenDataToWrite,
                          /*! ID of the device we're
                            writing for (differnet
                            devices may need to write
                            different pointers */
                          int deviceID,
                          /*! the geometry ID for which
                            we're generating the SBT
                            entry for */
                          int rayGenID,
                          /*! the raw void pointer the app has passed
                            during sbtGeomTypesBuild() */
                          const void *callBackUserData);
    
  /*! callback with which the app can specify what data is to be
    written into the SBT for a given geometry, ray type, and
    device */
  typedef void
  LLOWriteMissProgDataCB(uint8_t *missProgDataToWrite,
                         /*! ID of the device we're
                           writing for (differnet
                           devices may need to write
                           different pointers */
                         int deviceID,
                         /*! the ray type for which
                           we're generating the SBT
                           entry for */
                         int rayType,
                         /*! the raw void pointer the app has passed
                           during sbtMissProgsBuildd() */
                         const void *callBackUserData);
    

  
  /*! creates a new ll-owl device(group) context using the given CUDA
   *  device IDs. An empty list of device IDs is synonymous with "use
   *  all available device". If no context could be crated, the return
   *  value is null, and lloGetLastErrorText should contain an error
   *  message. */
  OWL_LL_INTERFACE
  LLOContext lloContextCreate(const int32_t *deviceIDs = nullptr,
                              int32_t        numDeviceIDs     = 0);
  
  OWL_LL_INTERFACE
  LLOResult lloContextDestroy(LLOContext llo);
  
  OWL_LL_INTERFACE
  LLOResult lloLaunch2D(LLOContext llo,
                        int32_t rayGenID,
                        int32_t launchDimX,
                        int32_t launchDimY);
  
  OWL_LL_INTERFACE
  LLOResult lloSetMaxInstancingDepth(LLOContext llo,
                                     int32_t maxInstanceDepth);
  
  OWL_LL_INTERFACE
  LLOResult lloAllocBuffers(LLOContext llo,
                            /*! number of buffers valid after this
                             *  function call */
                            int32_t numBuffers);

  OWL_LL_INTERFACE
  LLOResult lloAllocModules(LLOContext llo,
                            int        numModules);

  OWL_LL_INTERFACE
  LLOResult lloAllocMissProgs(LLOContext llo,
                              int        numMissProgs);

  OWL_LL_INTERFACE
  LLOResult lloAllocGroups(LLOContext llo,
                           int        numGroups);
  
  OWL_LL_INTERFACE
  LLOResult lloAllocGeoms(LLOContext llo,
                          int        numGeoms);

  OWL_LL_INTERFACE
  LLOResult lloAllocGeomTypes(LLOContext llo,
                              int        numGeomTypes);
  
  OWL_LL_INTERFACE
  LLOResult lloModuleCreate(LLOContext  llo,
                            int32_t     moduleID,
                            const char *ptxCode);

  /*! (re-)builds the modules that have been set via
   *  lloModuleCreate */
  OWL_LL_INTERFACE
  LLOResult lloBuildModules(LLOContext llo);
  

  OWL_LL_INTERFACE
  LLOResult lloAllocRayGens(LLOContext llo,
                            int32_t    rayGenProgCount);
  
  OWL_LL_INTERFACE
  LLOResult lloRayGenCreate(LLOContext  llo,
                            /*! ID of ray gen prog to create */
                            int32_t     programID,
                            /*! ID of module in which to look for that program */
                            int32_t     moduleID,
                            /*! name of the program */
                            const char *programName,
                            /*! size of that program's SBT data */
                            size_t      dataSizeOfRayGen);
  
  OWL_LL_INTERFACE
  LLOResult lloMissProgCreate(LLOContext  llo,
                              /*! ID of ray gen prog to create */
                              int32_t     programID,
                              /*! ID of module in which to look for that program */
                              int32_t     moduleID,
                              /*! name of the program */
                              const char *programName,
                              /*! size of that program's SBT data */
                              size_t      dataSizeOfMissProg);
  
  OWL_LL_INTERFACE
  LLOResult lloBuildPrograms(LLOContext llo);
  
  OWL_LL_INTERFACE
  LLOResult lloCreatePipeline(LLOContext llo);

  OWL_LL_INTERFACE
  LLOResult lloHostPinnedBufferCreate(LLOContext llo,
                                      /*! ID of buffer to create */
                                      int32_t    bufferID,
                                      /*! number of elements */
                                      size_t     sizeInBytes);

  OWL_LL_INTERFACE
  LLOResult lloDeviceBufferCreate(LLOContext  llo,
                                  /*! ID of buffer to create */
                                  int32_t     bufferID,
                                  /*! number of elements */
                                  size_t      sizeInBytes,
                                  const void *initData = nullptr);
  
  /*! builds the SBT's ray gen program entries, using the given
   *  callback to query the app as as to what values to write for a
   *  given ray gen program */
  OWL_LL_INTERFACE
  LLOResult lloSbtRayGensBuild(LLOContext           llo,
                               LLOWriteRayGenDataCB writeRayGenDataCB,
                               const void          *callBackData);

  /*! builds the SBT's miss program entries, using the given
   *  callback to query the app as as to what values to write for a
   *  given miss program */
  OWL_LL_INTERFACE
  LLOResult lloSbtMissProgsBuild(LLOContext             llo,
                                 LLOWriteMissProgDataCB writeMissProgDataCB,
                                 const void            *callBackData);

  /*! builds the SBT's hit program entries, using the given
   *  callback to query the app as as to what values to write for a
   *  given hit program */
  OWL_LL_INTERFACE
  LLOResult lloSbtHitProgsBuild(LLOContext           llo,
                                LLOWriteHitProgDataCB writeHitProgDataCB,
                                const void          *callBackData);
  
  OWL_LL_INTERFACE
  int32_t lloGetDeviceCount(LLOContext llo);
  
  /*! returns the device-side pointer of the given buffer, on the
   *  given device */
  OWL_LL_INTERFACE
  const void *lloBufferGetPointer(LLOContext llo,
                                  int32_t    bufferID,
                                  int32_t    deviceID);

  /*! returns the device-side pointer of the given buffer, on the
   *  given device */
  OWL_LL_INTERFACE
  OptixTraversableHandle lloGroupGetTraversable(LLOContext llo,
                                                int32_t    groupID,
                                                int32_t    deviceID);
  OWL_LL_INTERFACE
  uint32_t lloGroupGetSbtOffset(LLOContext llo,
                                int32_t    groupID);

  OWL_LL_INTERFACE
  LLOResult lloGeomTypeCreate(LLOContext llo,
                              int32_t geomTypeID,
                              size_t sizeOfSBTData);
  
  OWL_LL_INTERFACE
  LLOResult lloGeomTypeIntersect(LLOContext llo,
                                 int32_t geomTypeID,
                                 int32_t rayTypeID,
                                 int32_t moduleID,
                                 const char *programName);
  
  /*! set bounding box program for given geometry type, using a
    bounding box program to be called on the device. note that
    unlike other programs (intersect, closesthit, anyhit) these
    programs are not 'per ray type', but exist only once per
    geometry type. obviously only allowed for user geometry
    typed. */
  OWL_LL_INTERFACE
  LLOResult lloGeomTypeBoundsProgDevice(LLOContext llo,
                                        int32_t geomTypeID,
                                        int32_t moduleID,
                                        const char *programName,
                                        size_t geomDataSize);

  OWL_LL_INTERFACE
  LLOResult lloGeomTypeClosestHit(LLOContext llo,
                                  int32_t geomTypeID,
                                  int32_t rayTypeID,
                                  int32_t moduleID,
                                  const char *programName);
  OWL_LL_INTERFACE
  LLOResult lloGeomTypeAnyHit(LLOContext llo,
                              int32_t geomTypeID,
                              int32_t rayTypeID,
                              int32_t moduleID,
                              const char *programName);
  
  OWL_LL_INTERFACE
  LLOResult lloTrianglesGeomCreate(LLOContext llo,
                                   /*! ID of the geometry to create */
                                   int32_t    geomID,
                                   /*! ID of the geometry *type* to
                                       use for this geometry (this is
                                       what defines the SBT data size,
                                       closest hit program, etc */
                                   int32_t    geomTypeID);
  OWL_LL_INTERFACE
  LLOResult lloUserGeomCreate(LLOContext llo,
                              /*! ID of the geometry to create */
                              int32_t    geomID,
                              /*! ID of the geometry *type* to
                                use for this geometry (this is
                                what defines the SBT data size,
                                closest hit program, etc */
                              int32_t    geomTypeID,
                              size_t     numPrims);

      
  /*! set a buffer of bounding boxes that this user geometry will use
    when building the accel structure. this is one of multiple ways of
    specifying the bounding boxes for a user gometry (the other two
    being a) setting the geometry type's boundsFunc, or b) setting a
    host-callback fr computing the bounds). Only one of the three
    methods can be set at any given time */
  OWL_LL_INTERFACE
  LLOResult lloUserGeomSetBoundsBuffer(LLOContext llo,
                                       int32_t geomID,
                                       int32_t bufferID);

  OWL_LL_INTERFACE
  LLOResult lloUserGeomSetPrimCount(LLOContext llo,
                                    int32_t geomID,
                                    int32_t numPrims);
  
  OWL_LL_INTERFACE
  LLOResult lloInstanceGroupCreate(LLOContext     llo,
                                   int32_t        groupID,
                                   const int32_t *childGroupIDs,
                                   int32_t        numChildGroupIDs);
  OWL_LL_INTERFACE
  LLOResult lloTrianglesGeomGroupCreate(LLOContext     llo,
                                        int32_t        groupID,
                                        const int32_t *geomIDs,
                                        int32_t        numGeomIDs);
  OWL_LL_INTERFACE
  LLOResult lloUserGeomGroupCreate(LLOContext     llo,
                                   int32_t        groupID,
                                   const int32_t *geomIDs,
                                   int32_t        numGeomIDs);
  
  OWL_LL_INTERFACE
  LLOResult lloGroupAccelBuild(LLOContext llo,
                               int32_t    groupID);

  OWL_LL_INTERFACE
  LLOResult lloInstanceGroupSetTransform(LLOContext llo,
                                         int32_t    groupID,
                                         int32_t    childID,
                                         const float *xfm);
  
  OWL_LL_INTERFACE
  LLOResult lloInstanceGroupSetChild(LLOContext llo,
                                     int32_t    groupID,
                                     int32_t    childNo,
                                     int32_t    childGroupID,
                                     const float *xfm);
  
  OWL_LL_INTERFACE
  LLOResult lloGeomGroupSetChild(LLOContext llo,
                                 int32_t    groupID,
                                 int32_t    childNo,
                                 int32_t    childGeomID);
  
  OWL_LL_INTERFACE
  LLOResult lloGroupBuildPrimitiveBounds(LLOContext llo,
                                         int32_t    groupID,
                                         size_t     maxGeomDataSize,
                                         LLOWriteUserGeomBoundsDataCB cb,
                                         const void *cbData);
  
  OWL_LL_INTERFACE
  LLOResult lloTrianglesGeomSetVertexBuffer(LLOContext llo,
                                            int32_t    geomID,
                                            int32_t    bufferID,
      size_t    count,
                                            size_t     stride,
      size_t    offset);
  
  OWL_LL_INTERFACE
  LLOResult lloTrianglesGeomSetIndexBuffer(LLOContext llo,
                                           int32_t    geomID,
                                           int32_t    bufferID,
      size_t    count,
      size_t    stride,
      size_t    offset);
  
#ifdef __cplusplus
} // extern "C"
#endif

#ifdef __cplusplus

/*! C++-only wrapper of callback method with lambda function */
template<typename Lambda>
void lloSbtRayGensBuild(LLOContext llo,
                        const Lambda &l)
{
  lloSbtRayGensBuild
    (llo,
     [](uint8_t *output,
        int devID, int rgID, 
        const void *cbData)
     {
       const Lambda *lambda = (const Lambda *)cbData;
       (*lambda)(output,devID,rgID);
     },
     (const void *)&l);
}

/*! C++-only wrapper of callback method with lambda function */
template<typename Lambda>
void lloSbtHitProgsBuild(LLOContext llo,
                         const Lambda &l)
{
  lloSbtHitProgsBuild
    (llo,
     [](uint8_t *output,
        int devID,
        int geomID,
        int rayTypeID,
        const void *cbData)
     {
       const Lambda *lambda = (const Lambda *)cbData;
       (*lambda)(output,devID,geomID,rayTypeID);
     },
     (const void *)&l);
}

/*! C++-only wrapper of callback method with lambda function */
template<typename Lambda>
void lloSbtMissProgsBuild(LLOContext llo,
                          const Lambda &l)
{
  lloSbtMissProgsBuild
    (llo,
     [](uint8_t *output,
        int devID, int rgID, 
        const void *cbData)
     {
       const Lambda *lambda = (const Lambda *)cbData;
       (*lambda)(output,devID,rgID);
     },
     (const void *)&l);
}

/*! C++-only wrapper of callback method with lambda function */
template<typename Lambda>
void lloGroupBuildPrimitiveBounds(LLOContext llo,
                                  uint32_t groupID,
                                  size_t sizeOfData,
                                  const Lambda &l)
{
  lloGroupBuildPrimitiveBounds
    (llo,groupID,sizeOfData,
     [](uint8_t *output,
        int devID,
        int geomID,
        int childID, 
        const void *cbData)
     {
       const Lambda *lambda = (const Lambda *)cbData;
       (*lambda)(output,devID,geomID,childID);
     },
     (const void *)&l);
}
#endif



