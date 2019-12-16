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

#include "owl/ll/optix.h"

#define OWL_THROWS_EXCEPTIONS 1
#if OWL_THROWS_EXCEPTIONS
#define OWL_EXCEPT(a) throw std::runtime_error(a)
#else
#define OWL_EXCEPT(a) /* ignore */
#endif

namespace owl {
  namespace ll {

    typedef int32_t id_t;
    
    struct HostPinnedMemory
    {
      typedef std::shared_ptr<HostPinnedMemory> SP;
      HostPinnedMemory(size_t amount);
      ~HostPinnedMemory();
      void *pointer;
    };

    typedef void
    WriteUserGeomBoundsDataCB(uint8_t *userGeomDataToWrite,
                              int deviceID,
                              int geomID,
                              int childID,
                              const void *cbUserData);
    
    /*! callback with which the app can specify what data is to be
      written into the SBT for a given geometry, ray type, and
      device */
    typedef void
    WriteHitProgDataCB(uint8_t *hitProgDataToWrite,
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
    WriteRayGenDataCB(uint8_t *rayGenDataToWrite,
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
    WriteMissProgDataCB(uint8_t *missProgDataToWrite,
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
    
    struct Device;
    
    struct DeviceGroup {
      typedef std::shared_ptr<DeviceGroup> SP;

      DeviceGroup(const std::vector<Device *> &devices);
      ~DeviceGroup();

      /*! set the maximum instancing depth that will be allowed; '0'
          means 'no instancing, only bottom level accels', '1' means
          'only one singel level of instances' (ie, instancegroups
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
      /*! create a new module under given ID
       * 
       *  \todo rename to moduleCreate for consistency
       *
       *  \todo add module destroy
       *
       *  \warning deprecated naming */
      void setModule(size_t slot, const char *ptxCode);

      void moduleCreate(int moduleID, const char *ptxCode);
      void buildModules();
      void createPipeline();
      void buildPrograms();
      
      void allocGeomTypes(size_t count);
      void allocRayGens(size_t count);
      void allocMissProgs(size_t count);

      void geomTypeCreate(int geomTypeID,
                          size_t programDataSize);
                          
      /*! set bounding box program for given geometry type, using a
          bounding box program to be called on the device. note that
          unlike other programs (intersect, closesthit, anyhit) these
          programs are not 'per ray type', but exist only once per
          geometry type. obviously only allowed for user geometry
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
                          int numPrims);
      
      void trianglesGeomGroupCreate(int groupID,
                                    int *geomIDs, int geomCount);
      void userGeomGroupCreate(int groupID,
                               int *geomIDs, int geomCount);
      /*! create a new instance group with given list of children */
      void instanceGroupCreate(/*! the group we are defining */
                               int groupID,
                               /* list of children. list can be
                                  omitted by passing a nullptr, but if
                                  not null this must be a list of
                                  'childCount' valid group ID */
                               int *childGroupIDs,
                               /*! number of children in this group */
                               int childCount);
      /*! set given child's instance transform. groupID must be a
          valid instance group, childID must be wihtin
          [0..numChildren) */
      void instanceGroupSetTransform(int groupID,
                                     int childNo,
                                     const affine3f &xfm);
      /*! set given child to {childGroupID+xfm}  */
      void instanceGroupSetChild(int groupID,
                                 int childNo,
                                 int childGroupID,
                                 const affine3f &xfm=affine3f(gdt::one));
      void createDeviceBuffer(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData)
      {
        // TODO: ax this after renaming samples
        std::cout << "warning: deprecated, use deviceBufferCreate() instead" << std::endl;
        deviceBufferCreate(bufferID,elementCount,elementSize,initData);
      }

      /*! destroy the given buffer, and release all host and/or device
          memory associated with it */
      void bufferDestroy(int bufferID);
      
      /*! create a new device buffer - this buffer type will be
          allocated on each device */
      void deviceBufferCreate(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
      
      void createHostPinnedBuffer(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize)
      {
        // TODO: ax this after renaming samples
        std::cout << "warning: deprecated, use hostPinnedBufferCreate() instead" << std::endl;
        hostPinnedBufferCreate(bufferID,elementCount,elementSize);
      }
      void hostPinnedBufferCreate(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize);
      
      /*! returns the given device's buffer address on the specified
          device */
      void *bufferGetPointer(int bufferID, int devID);
      
      /*! set a buffer of bounding boxes that this user geometry will
          use when building the accel structure. this is one of
          multiple ways of specifying the bounding boxes for a user
          gometry (the other two being a) setting the geometry type's
          boundsFunc, or b) setting a host-callback fr computing the
          bounds). Only one of the three methods can be set at any
          given time */
      void userGeomSetBoundsBuffer(int geomID, int bufferID);

      void trianglesGeomSetVertexBuffer(int geomID,
                                        int bufferID,
                                        int count,
                                        int stride,
                                        int offset);
      void trianglesGeomSetIndexBuffer(int geomID,
                                       int bufferID,
                                       int count,
                                       int stride,
                                       int offset);
      void groupBuildAccel(int groupID);
      OptixTraversableHandle groupGetTraversable(int groupID, int deviceID);
      uint32_t groupGetSBTOffset(int groupID);


      void groupBuildPrimitiveBounds(int groupID,
                                     size_t maxGeomDataSize,
                                     WriteUserGeomBoundsDataCB cb,
                                     void *cbData);
      void sbtHitProgsBuild(WriteHitProgDataCB writeHitProgDataCB,
                            void *callBackData);
      void sbtRayGensBuild(WriteRayGenDataCB WriteRayGenDataCB,
                           void *callBackData);
      void sbtMissProgsBuild(WriteMissProgDataCB WriteMissProgDataCB,
                             void *callBackData);
      
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
          },(void *)&l);
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
                              },(void *)&l);
      }

      template<typename Lambda>
      void sbtRayGensBuild(const Lambda &l)
      {
        this->sbtRayGensBuild([](uint8_t *output,
                                 int devID, int rgID, 
                                 const void *cbData) {
                                const Lambda *lambda = (const Lambda *)cbData;
                                (*lambda)(output,devID,rgID);
                              },(void *)&l);
      }

      template<typename Lambda>
      void sbtMissProgsBuild(const Lambda &l)
      {
        this->sbtMissProgsBuild([](uint8_t *output,
                                   int devID, int rayType, 
                                   const void *cbData) {
                                  const Lambda *lambda = (const Lambda *)cbData;
                                  (*lambda)(output,devID,rayType);
                                },(void *)&l);
      }

      size_t getDeviceCount() const { return devices.size(); }
      void launch(int rgID, const vec2i &dims);

      
      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs. Note this is
         the only shared_ptr we use on that abstractoin level, but
         here we use one to force a proper destruction of the
         device */
      static DeviceGroup::SP create(const int *deviceIDs  = nullptr,
                                    size_t     numDevices = 0);
      static void destroy(DeviceGroup::SP &ll) { ll = nullptr; }

      /*! accessor helpers that first checks the validity of the given
        device ID, then returns the given device */
      Device *checkGetDevice(int deviceID);
      
      const std::vector<Device *> devices;
    };

  } // ::owl::ll
} //::owl
