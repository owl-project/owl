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

#include "ll/optix.h"
#ifdef __CUDA_ARCH__
#  error "this file should only ever get included on the device side"
#endif

namespace owl {
  namespace ll {

    struct HostPinnedMemory
    {
      typedef std::shared_ptr<HostPinnedMemory> SP;
      HostPinnedMemory(size_t amount);
      ~HostPinnedMemory();
      void *pointer;
    };
    
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
      
      void allocModules(size_t count);
      void setModule(size_t slot, const char *ptxCode);
      void buildModules();
      void createPipeline();
      void buildPrograms();
      
      void allocGeomTypes(size_t count);
      void allocRayGens(size_t count);
      void allocMissProgs(size_t count);

      void setGeomTypeClosestHit(int pgID, int rayTypeID, int moduleID, const char *progName);
      void setRayGen(int pgID, int moduleID, const char *progName);
      void setMissProg(int pgID, int moduleID, const char *progName);
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void reallocGroups(size_t newCount);
      void reallocBuffers(size_t newCount);
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void reallocGeoms(size_t newCount);

      void createTrianglesGeom(int geomID,
                               /*! the "logical" hit group ID:
                                 will always count 0,1,2... evne
                                 if we are using multiple ray
                                 types; the actual hit group
                                 used when building the SBT will
                                 then be 'geomTypeID *
                                 rayTypeCount) */
                               int geomTypeID);
      
      void createUserGeom(int geomID,
                          /*! the "logical" hit group ID:
                            will always count 0,1,2... evne
                            if we are using multiple ray
                            types; the actual hit group
                            used when building the SBT will
                            then be 'geomTypeID *
                            rayTypeCount) */
                          int geomTypeID,
                          int numPrims);
      
      void createTrianglesGeomGroup(int groupID,
                                    int *geomIDs, int geomCount);
      void createUserGeomGroup(int groupID,
                               int *geomIDs, int geomCount);

      void createDeviceBuffer(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
      void createHostPinnedBuffer(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize);
      
      /*! returns the given device's buffer address on the specified
          device */
      void *bufferGetPointer(int bufferID, int devID);
      
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
      
      void sbtGeomTypesBuild(size_t maxHitGroupDataSize,
                             WriteHitProgDataCB writeHitProgDataCB,
                             void *callBackData);
      void sbtRayGensBuild(size_t maxRayGenDataSize,
                           WriteRayGenDataCB WriteRayGenDataCB,
                           void *callBackData);
      void sbtMissProgsBuild(size_t maxMissProgDataSize,
                             WriteMissProgDataCB WriteMissProgDataCB,
                             void *callBackData);
      template<typename Lambda>
      void sbtGeomTypesBuild(size_t maxHitGroupDataSize,
                             const Lambda &l)
      {
        this->sbtGeomTypesBuild(maxHitGroupDataSize,
                              [](uint8_t *output,
                                 int devID,
                                 int geomID,
                                 int rayType,
                                 const void *cbData) {
                                const Lambda *lambda = (const Lambda *)cbData;
                                (*lambda)(output,devID,geomID,rayType,cbData);
                              },(void *)&l);
      }

      template<typename Lambda>
      void sbtRayGensBuild(size_t maxRayGenDataSize,
                           const Lambda &l)
      {
        this->sbtRayGensBuild(maxRayGenDataSize,
                              [](uint8_t *output,
                                 int devID, int rgID, 
                                 const void *cbData) {
                                const Lambda *lambda = (const Lambda *)cbData;
                                (*lambda)(output,devID,rgID,cbData);
                              },(void *)&l);
      }

      template<typename Lambda>
      void sbtMissProgsBuild(size_t maxMissProgDataSize,
                             const Lambda &l)
      {
        this->sbtMissProgsBuild(maxMissProgDataSize,
                                [](uint8_t *output,
                                   int devID, int rayType, 
                                   const void *cbData) {
                                  const Lambda *lambda = (const Lambda *)cbData;
                                  (*lambda)(output,devID,rayType,cbData);
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
