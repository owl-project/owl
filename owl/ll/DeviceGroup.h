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

namespace owl {
  namespace ll {

    /*! callback with which the app can specify what data is to be
      written into the SBT for a given geometry, ray type, and
      device */
    typedef void
    WriteHitGroupCallBack(uint8_t *hitGroupToWrite,
                          /*! ID of the device we're
                            writing for (differnet
                            devices may need to write
                            differnet pointers */
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
      
      void allocHitGroupPGs(size_t count);
      void allocRayGenPGs(size_t count);
      void allocMissPGs(size_t count);

      void setHitGroupClosestHit(int pgID, int moduleID, const char *progName);
      void setRayGenPG(int pgID, int moduleID, const char *progName);
      void setMissPG(int pgID, int moduleID, const char *progName);
      
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
                                 then be 'logicalHitGroupID *
                                 rayTypeCount) */
                               int logicalHitGroupID);
      
      void createTrianglesGeomGroup(int groupID,
                                    int *geomIDs, int geomCount);

      void createDeviceBuffer(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
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
      void sbtHitGroupsBuild(size_t maxHitGroupDataSize,
                             WriteHitGroupCallBack writeHitGroupCallBack,
                             void *callBackData);

      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs. Note this is
         the only shared_ptr we use on that abstractoin level, but
         here we use one to force a proper destruction of the
         device */
      static DeviceGroup::SP create(const int *deviceIDs  = nullptr,
                                    size_t     numDevices = 0);
      static void destroy(DeviceGroup::SP &ll) { ll = nullptr; }

      const std::vector<Device *> devices;
    };
    

  } // ::owl::ll
} //::owl
