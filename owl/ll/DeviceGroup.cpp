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

#define LOG(message)                            \
  std::cout << "#owl.ll: "                      \
  << message                                    \
  << std::endl

#define LOG_OK(message)                                 \
  std::cout << GDT_TERMINAL_LIGHT_GREEN                 \
  << "#owl.ll: "                                        \
  << message << GDT_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    HostPinnedMemory::HostPinnedMemory(size_t amount)
    {
      CUDA_CALL(MallocHost((void**)&pointer, amount));
      assert(pointer != nullptr);
    }
    
    HostPinnedMemory::~HostPinnedMemory()
    {
      assert(pointer != nullptr);
      CUDA_CALL(FreeHost(pointer));
      pointer = nullptr;
    }
    
    DeviceGroup::DeviceGroup(const std::vector<Device *> &devices)
      : devices(devices)
    {
      assert(!devices.empty());
    }

    DeviceGroup::~DeviceGroup()
    {
      LOG("destroying devices");
      for (auto device : devices) {
        assert(device);
        delete device;
      }
      LOG_OK("all devices properly destroyed");
    }

    /*! accessor helpers that first checks the validity of the given
      device ID, then returns the given device */
    Device *DeviceGroup::checkGetDevice(int deviceID)
    {
      assert("check valid device ID" && deviceID >= 0);
      assert("check valid device ID" && deviceID <  devices.size());
      Device *device = devices[deviceID];
      assert("check valid device" && device != nullptr);
      return device;
    }

    void DeviceGroup::allocModules(size_t count)
    { for (auto device : devices) device->allocModules(count); }
    
    void DeviceGroup::setModule(size_t slot, const char *ptxCode)
    { for (auto device : devices) device->modules.set(slot,ptxCode); }
    
    void DeviceGroup::buildModules()
    {
      for (auto device : devices)
        device->buildModules();
      LOG_OK("module(s) successfully (re-)built");
    }
    
    void DeviceGroup::allocGeomTypes(size_t count)
    { for (auto device : devices) device->allocGeomTypes(count); }
    
    void DeviceGroup::allocRayGens(size_t count)
    { for (auto device : devices) device->allocRayGens(count); }
    
    void DeviceGroup::allocMissProgs(size_t count)
    { for (auto device : devices) device->allocMissProgs(count); }

    /*! set bounding box program for given geometry type, using a
      bounding box program to be called on the device. note that
      unlike other programs (intersect, closesthit, anyhit) these
      programs are not 'per ray type', but exist only once per
      geometry type. obviously only allowed for user geometry
      typed. */
    void DeviceGroup::setGeomTypeBoundsProgDevice(int geomTypeID,
                                                  int moduleID,
                                                  const char *progName,
                                                  size_t geomDataSize)
    {
      for (auto device : devices)
        device->setGeomTypeBoundsProgDevice(geomTypeID,moduleID,progName,
                                            geomDataSize);
    }
    
      
    void DeviceGroup::setGeomTypeClosestHit(int geomTypeID,
                                            int rayTypeID,
                                            int moduleID,
                                            const char *progName)
    {
      for (auto device : devices)
        device->setGeomTypeClosestHit(geomTypeID,rayTypeID,moduleID,progName);
    }
    
    void DeviceGroup::setGeomTypeIntersect(int geomTypeID,
                                            int rayTypeID,
                                            int moduleID,
                                            const char *progName)
    {
      for (auto device : devices)
        device->setGeomTypeIntersect(geomTypeID,rayTypeID,moduleID,progName);
    }
    
    void DeviceGroup::setRayGen(int pgID,
                                int moduleID,
                                const char *progName,
                                size_t programDataSize)
    {
      for (auto device : devices)
        device->setRayGen(pgID,moduleID,progName,programDataSize);
    }
    
    /*! specifies which miss program to run for a given miss prog
      ID */
    void DeviceGroup::setMissProg(/*! miss program ID, in [0..numAllocatedMissProgs) */
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
                                  size_t missProgDataSize)
    {
      for (auto device : devices)
        device->setMissProg(programID,moduleID,progName,missProgDataSize);
    }

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

    void DeviceGroup::createUserGeom(int geomID,
                                     /*! the "logical" hit group ID:
                                       will always count 0,1,2... evne
                                       if we are using multiple ray
                                       types; the actual hit group
                                       used when building the SBT will
                                       then be 'logicalHitGroupID *
                                       rayTypeCount) */
                                     int logicalHitGroupID,
                                     int numPrims)
    {
      for (auto device : devices)
        device->createUserGeom(geomID,logicalHitGroupID,numPrims);
    }
      
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
                                               int *geomIDs,
                                               int geomCount)
    {
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && geomCount == 0) ||
              (geomIDs != nullptr && geomCount >  0)));
        
      for (auto device : devices) {
        device->createTrianglesGeomGroup(groupID,geomIDs,geomCount);
      }
    }

    void DeviceGroup::createUserGeomGroup(int groupID,
                                          int *geomIDs,
                                          int geomCount)
    {
      assert("check for valid combinations of child list" &&
             ((geomIDs == nullptr && geomCount == 0) ||
              (geomIDs != nullptr && geomCount >  0)));
        
      for (auto device : devices) {
        device->createUserGeomGroup(groupID,geomIDs,geomCount);
      }
    }

    void DeviceGroup::buildPrograms()
    {
      for (auto device : devices)
        device->buildPrograms();
      LOG_OK("device programs (re-)built");
    }
    
    void DeviceGroup::createPipeline()
    {
      for (auto device : devices)
        device->createPipeline();
      LOG_OK("optix pipeline created");
    }

    void DeviceGroup::createDeviceBuffer(int bufferID,
                                         size_t elementCount,
                                         size_t elementSize,
                                         const void *initData)
    {
      for (auto device : devices) 
        device->createDeviceBuffer(bufferID,elementCount,elementSize,initData);
    }

    void DeviceGroup::createHostPinnedBuffer(int bufferID,
                                             size_t elementCount,
                                             size_t elementSize)
    {
      HostPinnedMemory::SP pinned
        = std::make_shared<HostPinnedMemory>(elementCount*elementSize);
      for (auto device : devices) 
        device->createHostPinnedBuffer(bufferID,elementCount,elementSize,pinned);
    }

      
    /*! set a buffer of bounding boxes that this user geometry will
      use when building the accel structure. this is one of
      multiple ways of specifying the bounding boxes for a user
      gometry (the other two being a) setting the geometry type's
      boundsFunc, or b) setting a host-callback fr computing the
      bounds). Only one of the three methods can be set at any
      given time */
    void DeviceGroup::userGeomSetBoundsBuffer(int geomID,
                                              int bufferID)
    {
      for (auto device : devices) 
        device->userGeomSetBoundsBuffer(geomID,bufferID);
    }
    
    void DeviceGroup::trianglesGeomSetVertexBuffer(int geomID,
                                                   int bufferID,
                                                   int count,
                                                   int stride,
                                                   int offset)
    {
      for (auto device : devices) 
        device->trianglesGeomSetVertexBuffer(geomID,bufferID,count,stride,offset);
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
      try {
        for (auto device : devices) 
          device->groupBuildAccel(groupID);
      } catch (std::exception &e) {
        std::cerr << GDT_TERMINAL_RED
                  << "#owl.ll: Fatal error in owl::ll::groupBuildPrimitiveBounds():" << std::endl
                  << e.what()
                  << GDT_TERMINAL_DEFAULT << std::endl;
        exit(0);
      }
    }

    OptixTraversableHandle DeviceGroup::groupGetTraversable(int groupID, int deviceID)
    {
      return checkGetDevice(deviceID)->groupGetTraversable(groupID);
    }

    void DeviceGroup::sbtGeomTypesBuild(size_t maxHitGroupDataSize,
                                        WriteHitProgDataCB writeHitProgDataCB,
                                        void *callBackData)
    {
      for (auto device : devices) 
        device->sbtGeomTypesBuild(maxHitGroupDataSize,
                                  writeHitProgDataCB,
                                  callBackData);
    }
    
    void DeviceGroup::sbtRayGensBuild(WriteRayGenDataCB writeRayGenCB,
                                      void *callBackData)
    {
      for (auto device : devices) 
        device->sbtRayGensBuild(writeRayGenCB,
                                callBackData);
    }
    
    void DeviceGroup::sbtMissProgsBuild(WriteMissProgDataCB writeMissProgCB,
                                        void *callBackData)
    {
      for (auto device : devices) 
        device->sbtMissProgsBuild(writeMissProgCB,
                                  callBackData);
    }

    void DeviceGroup::groupBuildPrimitiveBounds(int groupID,
                                                size_t maxGeomDataSize,
                                                WriteUserGeomBoundsDataCB cb,
                                                void *cbData)
    {
      try {
        for (auto device : devices) 
          device->groupBuildPrimitiveBounds(groupID,
                                            maxGeomDataSize,
                                            cb,
                                            cbData);
      } catch (std::exception &e) {
        std::cerr << GDT_TERMINAL_RED
                  << "#owl.ll: Fatal error in owl::ll::groupBuildPrimitiveBounds():" << std::endl
                  << e.what()
                  << GDT_TERMINAL_DEFAULT << std::endl;
        exit(0);
      }
    }

    /*! returns the given device's buffer address on the specified
        device */
    void *DeviceGroup::bufferGetPointer(int bufferID, int devID)
    {
      return checkGetDevice(devID)->bufferGetPointer(bufferID);
    }
      
    void DeviceGroup::launch(int rgID, const vec2i &dims)
    {
      for (auto device : devices) device->launch(rgID,dims);
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
      LOG("initializing CUDA");
      cudaFree(0);
      
      int totalNumDevices = 0;
      cudaGetDeviceCount(&totalNumDevices);
      if (totalNumDevices == 0)
        throw std::runtime_error("#owl.ll: no CUDA capable devices found!");
      LOG_OK("found " << totalNumDevices << " CUDA device(s)");

      
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

      LOG_OK("successfully created device group with " << devices.size() << " devices");
      return std::make_shared<DeviceGroup>(devices);
    }

  } // ::owl::ll
} //::owl

