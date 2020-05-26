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

#define OWL_THROWS_EXCEPTIONS 1
#if OWL_THROWS_EXCEPTIONS
#define OWL_EXCEPT(a) throw std::runtime_error(a)
#else
#define OWL_EXCEPT(a) /* ignore */
#endif

namespace owl {
  namespace ll {

    typedef int32_t id_t;


    struct DeviceGroup;
    typedef DeviceGroup *LLOContext;
    
  typedef void
  (*LLOWriteUserGeomBoundsDataCB)(uint8_t *userGeomDataToWrite,
                                  int deviceID,
                                  int geomID,
                                  int childID,
                                  const void *cbUserData);
    
  typedef void
  (*LLOWriteLaunchParamsCB)(uint8_t *userGeomDataToWrite,
                            int deviceID,
                            const void *cbUserData);
    
  /*! callback with which the app can specify what data is to be
    written into the SBT for a given geometry, ray type, and
    device */
  typedef void
  (*LLOWriteHitProgDataCB)(uint8_t *hitProgDataToWrite,
                           /*! ID of the device we're
                             writing for (different
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
                            writing for (different
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
                           writing for (different
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

    /*! class that maintains the cuda host pinned memoery handle for a
        group of HostPinnedBuffer's (host pinned memory is shared
        across all devices) */
    struct HostPinnedMemory
    {
      typedef std::shared_ptr<HostPinnedMemory> SP;
      HostPinnedMemory(size_t amount);
      ~HostPinnedMemory();

      void free();
      void alloc(size_t newSizeInBytes);
      void *get() const { return pointer; }
      
      void *pointer = nullptr;
    };

    /*! class that maintains the cuda managed memoery address for a
        group of ManagedMemoryBuffer's (managed memory is allocated
        only once on the group level, and then shared across all
        devices) */
    struct ManagedMemory
    {
      typedef std::shared_ptr<ManagedMemory> SP;
      ManagedMemory(DeviceGroup *devGroup,
                    size_t amount,
                    /*! data with which to populate this buffer; may
                        be null, but has to be of size 'amount' if
                        not */
                    const void *initData);
      ~ManagedMemory();

      void free();
      void alloc(size_t newSizeInBytes);
      void *get() const { return pointer; }
      
      void *pointer = nullptr;
      DeviceGroup *devGroup;
    };

    struct Device;
    
    struct DeviceGroup {
      
      static int logging()
      {
#ifdef NDEBUG
        return false;
#else
        return true;
#endif
      }
      
      DeviceGroup(const std::vector<Device *> &devices);
      ~DeviceGroup();

      /*! set the maximum instancing depth that will be allowed; '0'
        means 'no instancing, only bottom level accels', '1' means
        'only one single level of instances' (i.e., instancegroups
        never have children that are themselves instance groups),
        etc. 

        Note we currently do *not* yet check the node graph as
        to whether it adheres to this value - if you use a node
        graph that's deeper than the value passed through this
        function you will most likely see optix crashing on you (and
        correctly so). See issue #1.

        Note this value will have to be set *before* the pipeline
        gets created */
      void setMaxInstancingDepth(int maxInstancingDepth);
      
      void allocModules(size_t count);
      void allocLaunchParams(size_t count);

      void moduleCreate(int moduleID, const char *ptxCode);
      void buildModules();
      void createPipeline();
      void buildPrograms();
      
      void allocGeomTypes(size_t count);
      void allocRayGens(size_t count);
      void allocMissProgs(size_t count);

      void geomTypeCreate(int geomTypeID,
                          size_t programDataSize);
      void launchParamsCreate(int launchParamsID,
                              size_t sizeOfData);
                          
      /*! Set bounding box program for given geometry type, using a
        bounding box program to be called on the device. Note that
        unlike other programs (intersect, closesthit, anyhit) these
        programs are not 'per ray type', but exist only once per
        geometry type. Obviously only allowed for user geometry
        typed. */
      void setGeomTypeBoundsProgDevice(int geomTypeID,
                                       int moduleID,
                                       const char *progName,
                                       size_t geomDataSize);
      void setGeomTypeProgramSize(int pgID,
                                  size_t );
      void setGeomTypeClosestHit(int pgID,
                                 int rayTypeID,
                                 int moduleID,
                                 const char *progName);
      void setGeomTypeAnyHit(int pgID,
                                 int rayTypeID,
                                 int moduleID,
                                 const char *progName);
      void setGeomTypeIntersect(int pgID,
                                int rayTypeID,
                                int moduleID,
                                const char *progName);
      void setRayGen(int pgID,
                     int moduleID,
                     const char *progName,
                     size_t programDataSize);

      /*! specifies which miss program to run for a given miss prog
        ID */
      void setMissProg(/*! miss program ID, in [0..numAllocatedMissProgs) */
                       int programID,
                       /*! ID of the module the program will be bound
                         in, in [0..numAllocedModules) */
                       int moduleID,
                       /*! name of the program. Note we do not NOT
                         create a copy of this string, so the string
                         has to remain valid for the duration of the
                         program */
                       const char *progName,
                       /*! size of that miss program's SBT data */
                       size_t missProgDataSize);
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void allocGroups(size_t newCount);
      void allocBuffers(size_t newCount);
      // void allocTextures(size_t newCount);
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void allocGeoms(size_t newCount);

      void trianglesGeomCreate(int geomID,
                               /*! the "logical" hit group ID:
                                 will always count 0,1,2... evne
                                 if we are using multiple ray
                                 types; the actual hit group
                                 used when building the SBT will
                                 then be 'geomTypeID *
                                 rayTypeCount) */
                               int geomTypeID);
      
      void userGeomCreate(int geomID,
                          /*! the "logical" hit group ID:
                            will always count 0,1,2... evne
                            if we are using multiple ray
                            types; the actual hit group
                            used when building the SBT will
                            then be 'geomTypeID *
                            rayTypeCount) */
                          int geomTypeID,
                          size_t numPrims);

      /*! create a new group object (with an associated BVH) over
        triangle geometries. All geomIDs in this group must be
        valid, and must refer to geometries of type TrianglesGeom.
          
        BVH does not get built until groupBuildAccel()

        \todo geomIDs may be null, in which case child geomeries may
        be set using geomGroupSetChild()
      */
      void trianglesGeomGroupCreate(int groupID,
                                    const int *geomIDs,
                                    size_t geomCount);
      void userGeomGroupCreate(int groupID,
                               const int *geomIDs,
                               size_t geomCount);
      /*! create a new instance group with given list of children */
      void instanceGroupCreate(/*! the group we are defining */
                               int groupID,
                               size_t numChildren,
                               /* list of children. list can be
                                  omitted by passing a nullptr, but if
                                  not null this must be a list of
                                  'childCount' valid group ID */
                               const uint32_t *childGroupIDs,
                               const uint32_t *instIDs,
                               const affine3f *xfms);
      /*! set given child's instance transform. groupID must be a
        valid instance group, childID must be wihtin
        [0..numChildren) */
      void instanceGroupSetTransform(int groupID,
                                     int childNo,
                                     const affine3f &xfm);
      /*! set given child to {childGroupID+xfm}  */
      void instanceGroupSetChild(int groupID,
                                 int childNo,
                                 int childGroupID);
      void geomGroupSetChild(int groupID,
                             int childNo,
                             int childID);

      /*! destroy the given buffer, and release all host and/or device
        memory associated with it */
      void bufferDestroy(int bufferID);
      
      /*! create a new device buffer - this buffer type will be
        allocated on each device */
      void deviceBufferCreate(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
      
      /*! create a host-pinned memory buffer */
      void hostPinnedBufferCreate(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize);

      /*! create a managed memory buffer */
      void managedMemoryBufferCreate(int bufferID,
                                     size_t elementCount,
                                     size_t elementSize,
                                     const void *initData);

      void graphicsBufferCreate(int bufferID,
                                size_t elementCount,
                                size_t elementSize,
                                cudaGraphicsResource_t resource);

      void graphicsBufferMap(int bufferID);

      void graphicsBufferUnmap(int bufferID);
      
      void bufferResize(int bufferID, size_t newItemCount);
      void bufferUpload(int bufferID, const void *hostPtr);
      
      /*! returns the given device's buffer address on the specified
        device */
      void *bufferGetPointer(int bufferID, int devID);
      
      /*! return the cuda stream by the given launchparams object, on
        given device */
      cudaStream_t launchParamsGetStream(int launchParamsID, int devID);
      
      /*! Set a buffer of bounding boxes that this user geometry will
        use when building the accel structure. This is one of
        multiple ways of specifying the bounding boxes for a user
        geometry (the other two being a) setting the geometry type's
        boundsFunc, or b) setting a host-callback fr computing the
        bounds). Only one of the three methods can be set at any
        given time. */
      void userGeomSetBoundsBuffer(int geomID, int bufferID);
      void userGeomSetPrimCount(int geomID,
                                size_t numPrims);

      void trianglesGeomSetVertexBuffer(int geomID,
                                        int bufferID,
          size_t count,
          size_t stride,
          size_t offset);
      void trianglesGeomSetIndexBuffer(int geomID,
                                       int bufferID,
          size_t count,
          size_t stride,
          size_t offset);
      void groupBuildAccel(int groupID);
      // NEW naming:
      void groupAccelBuild(int groupID) { groupBuildAccel(groupID); }
      OptixTraversableHandle groupGetTraversable(int groupID, int deviceID);
      uint32_t groupGetSBTOffset(int groupID);


      void groupBuildPrimitiveBounds(int groupID,
                                     size_t maxGeomDataSize,
                                     LLOWriteUserGeomBoundsDataCB cb,
                                     const void *cbData);
      void sbtHitProgsBuild(LLOWriteHitProgDataCB writeHitProgDataCB,
                            const void *callBackData);
      void sbtRayGensBuild(LLOWriteRayGenDataCB WriteRayGenDataCB,
                           const void *callBackData);
      void sbtMissProgsBuild(LLOWriteMissProgDataCB WriteMissProgDataCB,
                             const void *callBackData);
      
      template<typename Lambda>
      void groupBuildPrimitiveBounds(int groupID,
                                     size_t maxGeomDataSize,
                                     const Lambda &l)
      {
        groupBuildPrimitiveBounds
          (groupID,maxGeomDataSize,
           [](uint8_t *output,
              int devID,
              int geomID,
              int childID,
              const void *cbData) {
            const Lambda *lambda = (const Lambda *)cbData;
            (*lambda)(output,devID,geomID,childID);
          },(const void *)&l);
      }
      
      
      template<typename Lambda>
      void sbtHitProgsBuild(const Lambda &l)
      {
        this->sbtHitProgsBuild([](uint8_t *output,
                                  int devID,
                                  int geomID,
                                  int childID,
                                  const void *cbData) {
                                 const Lambda *lambda = (const Lambda *)cbData;
                                 (*lambda)(output,devID,geomID,childID);
                               },(const void *)&l);
      }

      template<typename Lambda>
      void sbtRayGensBuild(const Lambda &l)
      {
        this->sbtRayGensBuild([](uint8_t *output,
                                 int devID, int rgID, 
                                 const void *cbData) {
                                const Lambda *lambda = (const Lambda *)cbData;
                                (*lambda)(output,devID,rgID);
                              },(const void *)&l);
      }

      template<typename Lambda>
      void sbtMissProgsBuild(const Lambda &l)
      {
        this->sbtMissProgsBuild([](uint8_t *output,
                                   int devID, int rayType, 
                                   const void *cbData) {
                                  const Lambda *lambda = (const Lambda *)cbData;
                                  (*lambda)(output,devID,rayType);
                                },(const void *)&l);
      }

      size_t getDeviceCount() const { return devices.size(); }
      void setRayTypeCount(size_t rayTypeCount);
      
      void launch(int rgID,
                  const vec2i &dims);

      void launch(int rgID,
                  const vec2i &dims,
                  int32_t launchParamsID,
                  LLOWriteLaunchParamsCB writeLaunchParamsCB,
                  const void *cbData);
      
      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs. */
      static DeviceGroup *create(const int *deviceIDs  = nullptr,
                                 size_t     numDevices = 0);
      static void destroy(DeviceGroup *&ll) { delete ll; ll = nullptr; }

      /*! accessor helpers that first checks the validity of the given
        device ID, then returns the given device */
      Device *checkGetDevice(int deviceID);

      /*! helper function that enables peer access across all devices */
      void enablePeerAccess();

      const std::vector<Device *> devices;
    };

  } // ::owl::ll
} //::owl
