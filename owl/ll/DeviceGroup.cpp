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

#include "ll/Device.h"
#include "ll/DeviceGroup.h"

#define GLOG(message)                           \
  std::cout << "#owl.ll: "                      \
  << message                                    \
  << std::endl

#define GLOG_OK(message)                                \
  std::cout << GDT_TERMINAL_GREEN                       \
  << "#owl.ll: "                                        \
  << message << GDT_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {
    
    DeviceGroup::DeviceGroup(const std::vector<Device *> &devices)
      : devices(devices)
    {
      assert(!devices.empty());
      GLOG_OK("successfully created device group with "
              << devices.size() << " device(s)");
    }
    
    DeviceGroup::~DeviceGroup()
    {
      GLOG("destroying devices");
      for (auto device : devices) {
        assert(device);
        delete device;
      }
      GLOG("all devices properly destroyed");
    }

    void DeviceGroup::allocModules(size_t count)
    { for (auto device : devices) device->allocModules(count); }
    
    void DeviceGroup::setModule(size_t slot, const char *ptxCode)
    { for (auto device : devices) device->modules.set(slot,ptxCode); }
    
    void DeviceGroup::buildModules()
    {
      for (auto device : devices)
        device->buildModules();
    }
    
    void DeviceGroup::allocHitGroupPGs(size_t count)
    { for (auto device : devices) device->allocHitGroupPGs(count); }
    
    void DeviceGroup::allocRayGenPGs(size_t count)
    { for (auto device : devices) device->allocRayGenPGs(count); }
    
    void DeviceGroup::allocMissPGs(size_t count)
    { for (auto device : devices) device->allocMissPGs(count); }

    void DeviceGroup::setHitGroupClosestHit(int pgID, int moduleID, const char *progName)
    { for (auto device : devices) device->setHitGroupClosestHit(pgID,moduleID,progName); }
    
    void DeviceGroup::setRayGenPG(int pgID, int moduleID, const char *progName)
    { for (auto device : devices) device->setRayGenPG(pgID,moduleID,progName); }
    
    void DeviceGroup::setMissPG(int pgID, int moduleID, const char *progName)
    { for (auto device : devices) device->setMissPG(pgID,moduleID,progName); }

    /*! resize the array of geom IDs. this can be either a
      'grow' or a 'shrink', but 'shrink' is only allowed if all
      geoms that would get 'lost' have alreay been
      destroyed */
    void DeviceGroup::reallocGroups(size_t newCount)
    { for (auto device : devices) device->reallocGroups(newCount); }
      
    void DeviceGroup::reallocBuffers(size_t newCount)
    { for (auto device : devices) device->reallocBuffers(newCount); }
      
    void DeviceGroup::reallocGeoms(size_t newCount)
    { for (auto device : devices) device->reallocGeoms(newCount); }

    void DeviceGroup::createTrianglesGeom(int geomID,
                                          /*! the "logical" hit group ID:
                                            will always count 0,1,2... evne
                                            if we are using multiple ray
                                            types; the actual hit group
                                            used when building the SBT will
                                            then be 'logicalHitGroupID *
                                            rayTypeCount) */
                                          int logicalHitGroupID)
    {
      for (auto device : devices)
        device->createTrianglesGeom(geomID,logicalHitGroupID);
    }

    void DeviceGroup::createTrianglesGeomGroup(int groupID,
                                               int *geomIDs, int geomCount)
    {
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && geomCount == 0) ||
              (geomIDs != nullptr && geomCount >  0)));
        
      for (auto device : devices) {
        device->createTrianglesGeomGroup(groupID,geomIDs,geomCount);
      }
    }

    void DeviceGroup::buildPrograms()
    {
      for (auto device : devices)
        device->buildPrograms();
      GLOG_OK("device programs (re-)built");
    }
    
    void DeviceGroup::createPipeline()
    {
      for (auto device : devices)
        device->createPipeline();
      GLOG_OK("optix pipeline created");
    }

    void DeviceGroup::createDeviceBuffer(int bufferID,
                                         size_t elementCount,
                                         size_t elementSize,
                                         const void *initData)
    {
      for (auto device : devices) {
        device->createDeviceBuffer(bufferID,elementCount,elementSize,initData);
      }
    }

    void DeviceGroup::trianglesGeomSetVertexBuffer(int geomID,
                                                   int bufferID,
                                                   int count,
                                                   int stride,
                                                   int offset)
    {
      for (auto device : devices) {
        device->trianglesGeomSetVertexBuffer(geomID,bufferID,count,stride,offset);
      }
    }
    
    void DeviceGroup::trianglesGeomSetIndexBuffer(int geomID,
                                                  int bufferID,
                                                  int count,
                                                  int stride,
                                                  int offset)
    {
      for (auto device : devices) {
        device->trianglesGeomSetIndexBuffer(geomID,bufferID,count,stride,offset);
      }
    }

    void DeviceGroup::groupBuildAccel(int groupID)
    {
      for (auto device : devices) 
        device->groupBuildAccel(groupID);
    }
    
    void DeviceGroup::sbtHitGroupsBuild(size_t maxHitGroupDataSize,
                                        WriteHitGroupCallBack writeHitGroupCallBack,
                                        void *callBackData)
    {
      for (auto device : devices) 
        device->sbtHitGroupsBuild(maxHitGroupDataSize,
                                  writeHitGroupCallBack,
                                  callBackData);
    }
    
    /* create an instance of this object that has properly
       initialized devices */
    DeviceGroup::SP DeviceGroup::create(const int *deviceIDs,
                                        size_t     numDevices)
    {
      assert((deviceIDs == nullptr && numDevices == 0)
             ||
             (deviceIDs != nullptr && numDevices > 0));

      // ------------------------------------------------------------------
      // init cuda, and error-out if no cuda devices exist
      // ------------------------------------------------------------------
      GLOG("initializing CUDA");
      cudaFree(0);
      
      int totalNumDevices = 0;
      cudaGetDeviceCount(&totalNumDevices);
      if (totalNumDevices == 0)
        throw std::runtime_error("#owl.ll: no CUDA capable devices found!");
      std::cout << "#owl.ll: found " << totalNumDevices << " CUDA device(s)" << std::endl;

      
      // ------------------------------------------------------------------
      // init optix itself
      // ------------------------------------------------------------------
      std::cout << "#owl.ll: initializing optix 7" << std::endl;
      OPTIX_CHECK(optixInit());

      // ------------------------------------------------------------------
      // check if a device ID list was passed, and if not, create one
      // ------------------------------------------------------------------
      std::vector<int> allDeviceIDs;
      if (deviceIDs == nullptr) {
        for (int i=0;i<totalNumDevices;i++) allDeviceIDs.push_back(i);
        numDevices = allDeviceIDs.size();
        deviceIDs  = allDeviceIDs.data();
      }
      // from here on, we need a non-empty list of requested device IDs
      assert(deviceIDs != nullptr && numDevices > 0);
      
      // ------------------------------------------------------------------
      // create actual devices, ignoring those that failed to initialize
      // ------------------------------------------------------------------
      std::vector<Device *> devices;
      for (int i=0;i<numDevices;i++) {
        try {
          Device *dev = new Device((int)devices.size(),deviceIDs[i]);
          assert(dev);
          devices.push_back(dev);
        } catch (std::exception &e) {
          std::cout << GDT_TERMINAL_RED
                    << "#owl.ll: Error creating optix device on CUDA device #"
                    << deviceIDs[i] << ": " << e.what() << " ... dropping this device"
                    << GDT_TERMINAL_DEFAULT << std::endl;
        }
      }

      // ------------------------------------------------------------------
      // some final sanity check that we managed to create at least
      // one device...
      // ------------------------------------------------------------------
      if (devices.empty())
        throw std::runtime_error("fatal error - could not find/create any optix devices");
      
      return std::make_shared<DeviceGroup>(devices);
    }

  } // ::owl::ll
} //::owl

