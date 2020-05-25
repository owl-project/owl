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

#include "owl/ll/Device.h"
#include "owl/ll/DeviceGroup.h"

#define LOG(message)                            \
  if (Context::logging())                       \
    std::cout << "#owl.ll: "                    \
              << message                        \
              << std::endl

#define LOG_OK(message)                                 \
  if (Context::logging())                               \
    std::cout << OWL_TERMINAL_LIGHT_GREEN               \
              << "#owl.ll: "                            \
              << message << OWL_TERMINAL_DEFAULT << std::endl

namespace owl {
  namespace ll {

    // ##################################################################
    // HostPinnedMemory
    // ##################################################################

    HostPinnedMemory::HostPinnedMemory(size_t amount)
    {
      alloc(amount);
      assert(pointer != nullptr);
    }
    
    HostPinnedMemory::~HostPinnedMemory()
    {
      assert(pointer != nullptr);
      free();
    }

    void HostPinnedMemory::alloc(size_t amount)
    {
      CUDA_CALL(MallocHost((void**)&pointer, amount));
    }

    void HostPinnedMemory::free()
    {
      CUDA_CALL_NOTHROW(FreeHost(pointer));
      pointer = nullptr;
    }


    // ##################################################################
    // ManagedMemoryMemory
    // ##################################################################

    ManagedMemory::ManagedMemory(DeviceGroup *devGroup,
                                 size_t amount,
                                 /*! data with which to populate this buffer; may
                                   be null, but has to be of size 'amount' if
                                   not */
                                 const void *initData)
      : devGroup(devGroup)
    {
      alloc(amount);
      if (initData)
        CUDA_CALL(Memcpy(pointer,initData,amount,
                         cudaMemcpyDefault));
      assert(pointer != nullptr);
    }
    
    ManagedMemory::~ManagedMemory()
    {
      assert(pointer != nullptr);
      free();
    }

    void ManagedMemory::alloc(size_t amount)
    {
      CUDA_CALL(MallocManaged((void**)&pointer, amount));
      // CUDA_CALL(MemAdvise((void*)pointer, amount, cudaMemAdviseSetReadMostly, -1));
      unsigned char *mem_end = (unsigned char *)pointer + amount;
      size_t pageSize = 16*1024*1024;
      int pageID = 0;
      for (unsigned char *begin = (unsigned char *)pointer; begin < mem_end; begin += pageSize) {
        unsigned char *end = std::min(begin+pageSize,mem_end);
        int devID = pageID++ % devGroup->devices.size();
        int cudaDevID = devGroup->devices[devID]->getCudaDeviceID();
        int result = 0;
        cudaDeviceGetAttribute (&result, cudaDevAttrConcurrentManagedAccess, cudaDevID);
        if (result) {
          CUDA_CALL(MemAdvise((void*)begin, end-begin, cudaMemAdviseSetPreferredLocation, cudaDevID));
        }
      }
    }
    
    void ManagedMemory::free()
    {
      CUDA_CALL_NOTHROW(Free(pointer));
      pointer = nullptr;
    }




    // ##################################################################
    // Device Group
    // ##################################################################

    DeviceGroup::DeviceGroup(const std::vector<Device *> &devices)
      : devices(devices)
    {
      assert(!devices.empty());
      enablePeerAccess();
    }


    void DeviceGroup::enablePeerAccess()
    {
      LOG("enabling peer access ('.'=self, '+'=can access other device)");
      int restoreActiveDevice = -1;
      cudaGetDevice(&restoreActiveDevice);
      int deviceCount = devices.size();
      LOG("found " << deviceCount << " CUDA capable devices");
      for (int i=0;i<deviceCount;i++) {
        LOG(" - device #" << i << " : " << devices[i]->getDeviceName());
      }
      LOG("enabling peer access:");
      for (int i=0;i<deviceCount;i++) {
        std::stringstream ss;
        ss << " - device #" << i << " : ";
        int cuda_i = devices[i]->getCudaDeviceID();
        for (int j=0;j<deviceCount;j++) {
          if (j == i) {
            ss << " ."; 
          } else {
            int cuda_j = devices[j]->getCudaDeviceID();
            int canAccessPeer = 0;
            cudaError_t rc = cudaDeviceCanAccessPeer(&canAccessPeer, cuda_i,cuda_j);
            if (rc != cudaSuccess)
              throw std::runtime_error("cuda error in cudaDeviceCanAccessPeer: "+std::to_string(rc));
            if (!canAccessPeer)
              throw std::runtime_error("could not enable peer access!?");
            
            cudaSetDevice(cuda_i);
            rc = cudaDeviceEnablePeerAccess(cuda_j,0);
            if (rc != cudaSuccess)
              throw std::runtime_error("cuda error in cudaDeviceEnablePeerAccess: "+std::to_string(rc));
            ss << " +";
          }
        }
        LOG(ss.str()); 
     }
      cudaSetDevice(restoreActiveDevice);
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
    void DeviceGroup::setMaxInstancingDepth(int maxInstancingDepth)
    {
      for (auto device : devices)
        device->setMaxInstancingDepth(maxInstancingDepth);
    }

    void DeviceGroup::setRayTypeCount(size_t rayTypeCount)
    {
      for (auto device : devices)
        device->setRayTypeCount(rayTypeCount);
    }


    void DeviceGroup::allocModules(size_t count)
    { for (auto device : devices) device->allocModules(count); }

    void DeviceGroup::allocLaunchParams(size_t count)
    { for (auto device : devices) device->allocLaunchParams(count); }
    
    void DeviceGroup::moduleCreate(int moduleID, const char *ptxCode)
    { for (auto device : devices) device->modules.set(moduleID,ptxCode); }

    void DeviceGroup::launchParamsCreate(int launchParamsID, size_t sizeOfVars)
    {
      for (auto device : devices)
        device->launchParamsCreate(launchParamsID,sizeOfVars);
    }

    
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

    /*! Set bounding box program for given geometry type, using a
      bounding box program to be called on the device. Note that
      unlike other programs (intersect, closesthit, anyhit) these
      programs are not 'per ray type', but exist only once per
      geometry type. Obviously only allowed for user geometry
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
    
    void DeviceGroup::setGeomTypeAnyHit(int geomTypeID,
                                            int rayTypeID,
                                            int moduleID,
                                            const char *progName)
    {
      for (auto device : devices)
        device->setGeomTypeAnyHit(geomTypeID,rayTypeID,moduleID,progName);
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
    void DeviceGroup::allocGroups(size_t newCount)
    {
      for (auto device : devices)
        device->allocGroups(newCount);
    }
      
    void DeviceGroup::allocBuffers(size_t newCount)
    {
      for (auto device : devices)
        device->allocBuffers(newCount);
    }
      
    // void DeviceGroup::allocTextures(size_t newCount)
    // {
    //   for (auto device : devices)
    //     device->allocTextures(newCount);
    // }
      
    void DeviceGroup::allocGeoms(size_t newCount)
    {
      for (auto device : devices)
        device->allocGeoms(newCount);
    }

    void DeviceGroup::userGeomCreate(int geomID,
                                     /*! the "logical" hit group ID:
                                       will always count 0,1,2... evne
                                       if we are using multiple ray
                                       types; the actual hit group
                                       used when building the SBT will
                                       then be 'logicalHitGroupID *
                                       rayTypeCount) */
                                     int logicalHitGroupID,
                                     size_t numPrims)
    {
      for (auto device : devices)
        device->userGeomCreate(geomID,logicalHitGroupID,numPrims);
    }
      
    void DeviceGroup::trianglesGeomCreate(int geomID,
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
        device->trianglesGeomCreate(geomID,logicalHitGroupID);
    }

    void DeviceGroup::trianglesGeomGroupCreate(int groupID,
                                               const int *geomIDs,
                                               size_t geomCount)
    {
      for (auto device : devices) {
        device->trianglesGeomGroupCreate(groupID,geomIDs,geomCount);
      }
    }

    void DeviceGroup::userGeomGroupCreate(int groupID,
                                          const int *geomIDs,
                                          size_t geomCount)
    {
      for (auto device : devices) {
        device->userGeomGroupCreate(groupID,geomIDs,geomCount);
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

    void DeviceGroup::bufferDestroy(int bufferID)
    {
      for (auto device : devices) 
        device->bufferDestroy(bufferID);
    }

    void DeviceGroup::deviceBufferCreate(int bufferID,
                                         size_t elementCount,
                                         size_t elementSize,
                                         const void *initData)
    {
      for (auto device : devices) 
        device->deviceBufferCreate(bufferID,elementCount,elementSize,initData);
    }

    void DeviceGroup::hostPinnedBufferCreate(int bufferID,
                                             size_t elementCount,
                                             size_t elementSize)
    {
      HostPinnedMemory::SP pinned
        = std::make_shared<HostPinnedMemory>(elementCount*elementSize);
      for (auto device : devices) 
        device->hostPinnedBufferCreate(bufferID,elementCount,elementSize,pinned);
    }

    void DeviceGroup::managedMemoryBufferCreate(int bufferID,
                                                size_t elementCount,
                                                size_t elementSize,
                                                const void *initData)
    {
      ManagedMemory::SP mem
        = std::make_shared<ManagedMemory>(this,elementCount*elementSize,initData);
      for (auto device : devices) 
        device->managedMemoryBufferCreate(bufferID,elementCount,elementSize,mem);
    }

    void DeviceGroup::graphicsBufferCreate(int bufferID,
                                           size_t elementCount,
                                           size_t elementSize,
                                           cudaGraphicsResource_t resource)
    {
      for (auto device : devices)
        device->graphicsBufferCreate(bufferID, elementCount, elementSize, resource);
    }

    void DeviceGroup::graphicsBufferMap(int bufferID)
    {
      for (auto device : devices)
        device->graphicsBufferMap(bufferID);
    }

    void DeviceGroup::graphicsBufferUnmap(int bufferID)
    {
      for (auto device : devices)
        device->graphicsBufferUnmap(bufferID);
    }
      
    /*! Set a buffer of bounding boxes that this user geometry will
      use when building the accel structure. This is one of
      multiple ways of specifying the bounding boxes for a user
      geometry (the other two being a) setting the geometry type's
      boundsFunc, or b) setting a host-callback fr computing the
      bounds). Only one of the three methods can be set at any
      given time. */
    void DeviceGroup::userGeomSetBoundsBuffer(int geomID,
                                              int bufferID)
    {
      for (auto device : devices) 
        device->userGeomSetBoundsBuffer(geomID,bufferID);
    }
    
    void DeviceGroup::userGeomSetPrimCount(int geomID,
                                           size_t count)
    {
      for (auto device : devices) 
        device->userGeomSetPrimCount(geomID,count);
    }
    
    void DeviceGroup::trianglesGeomSetVertexBuffer(int geomID,
                                                   int bufferID,
                                                   size_t count,
        size_t stride,
        size_t offset)
    {
      for (auto device : devices) 
        device->trianglesGeomSetVertexBuffer(geomID,bufferID,count,stride,offset);
    }
    
    void DeviceGroup::trianglesGeomSetIndexBuffer(int geomID,
                                                  int bufferID,
        size_t count,
        size_t stride,
        size_t offset)
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
        std::cerr << OWL_TERMINAL_RED
                  << "#owl.ll: Fatal error in owl::ll::groupBuildPrimitiveBounds():" << std::endl
                  << e.what()
                  << OWL_TERMINAL_DEFAULT << std::endl;
        throw e;
      }
    }

    uint32_t DeviceGroup::groupGetSBTOffset(int groupID)
    {
      return devices[0]->groupGetSBTOffset(groupID);
    }

    OptixTraversableHandle DeviceGroup::groupGetTraversable(int groupID, int deviceID)
    {
      return checkGetDevice(deviceID)->groupGetTraversable(groupID);
    }

    void DeviceGroup::sbtHitProgsBuild(LLOWriteHitProgDataCB writeHitProgDataCB,
                                       const void *callBackData)
    {
      for (auto device : devices) 
        device->sbtHitProgsBuild(writeHitProgDataCB,
                                 callBackData);
    }

    void DeviceGroup::geomTypeCreate(int geomTypeID,
                                     size_t programDataSize)
    {
      for (auto device : devices) 
        device->geomTypeCreate(geomTypeID,
                               programDataSize);
    }
                          

    void DeviceGroup::sbtRayGensBuild(LLOWriteRayGenDataCB writeRayGenCB,
                                      const void *callBackData)
    {
      for (auto device : devices) 
        device->sbtRayGensBuild(writeRayGenCB,
                                callBackData);
    }
    
    void DeviceGroup::sbtMissProgsBuild(LLOWriteMissProgDataCB writeMissProgCB,
                                        const void *callBackData)
    {
      for (auto device : devices) 
        device->sbtMissProgsBuild(writeMissProgCB,
                                  callBackData);
    }

    void DeviceGroup::groupBuildPrimitiveBounds(int groupID,
                                                size_t maxGeomDataSize,
                                                LLOWriteUserGeomBoundsDataCB cb,
                                                const void *cbData)
    {
        for (auto device : devices) 
          device->groupBuildPrimitiveBounds(groupID,
                                            maxGeomDataSize,
                                            cb,
                                            cbData);
    }

    
    /*! set given child's instance transform. groupID must be a
      valid instance group, childID must be wihtin
      [0..numChildren) */
    void DeviceGroup::instanceGroupSetTransform(int groupID,
                                                int childNo,
                                                const affine3f &xfm)
    {
      for (auto device : devices)
        device->instanceGroupSetTransform(groupID,
                                          childNo,
                                          xfm);
    }

    /*! set given child to {childGroupID+xfm}  */
    void DeviceGroup::instanceGroupSetChild(int groupID,
                                            int childNo,
                                            int childGroupID)
    {
      for (auto device : devices)
        device->instanceGroupSetChild(groupID,
                                      childNo,
                                      childGroupID);
    }

    void DeviceGroup::bufferResize(int bufferID, size_t newItemCount)
    {
      for (auto device : devices)
        device->bufferResize(bufferID,newItemCount);
    }
    
    void DeviceGroup::bufferUpload(int bufferID, const void *hostPtr)
    {
      for (auto device : devices)
        device->bufferUpload(bufferID,hostPtr);
    }
      

    void DeviceGroup::geomGroupSetChild(int groupID,
                                        int childNo,
                                        int childID)
    {
      for (auto device : devices)
        device->geomGroupSetChild(groupID,
                                  childNo,
                                  childID);
    }

    /*! create a new instance group with given list of children */
    void DeviceGroup::instanceGroupCreate(/*! the group we are defining */
                                          int groupID,
                                          size_t numChildren,
                                          /* list of children. list can be
                                             omitted by passing a nullptr, but if
                                             not null this must be a list of
                                             'childCount' valid group ID */
                                          const uint32_t *childGroupIDs,
                                          const uint32_t *instIDs,
                                          const affine3f *xfms)
    {
      for (auto device : devices)
        device->instanceGroupCreate(groupID,numChildren,
                                    childGroupIDs,instIDs,xfms);
    }

    /*! returns the given device's buffer address on the specified
        device */
    void *DeviceGroup::bufferGetPointer(int bufferID, int devID)
    {
      return checkGetDevice(devID)->bufferGetPointer(bufferID);
    }
    
    /*! return the cuda stream by the given launchparams object, on
      given device */
    CUstream DeviceGroup::launchParamsGetStream(int launchParamsID, int devID)
    {
      return checkGetDevice(devID)->launchParamsGetStream(launchParamsID);
    }
    
      
    void DeviceGroup::launch(int rgID, const vec2i &dims)
    {
      for (auto device : devices) device->launch(rgID,dims);
      CUDA_SYNC_CHECK();
    }
    
    void DeviceGroup::launch(int rgID,
                             const vec2i &dims,
                             int32_t launchParamsID,
                             LLOWriteLaunchParamsCB writeLaunchParamsCB,
                             const void *cbData)
    {
      for (auto device : devices)
        device->launch(rgID,dims,
                       launchParamsID,
                       writeLaunchParamsCB,
                       cbData);
      // CUDA_SYNC_CHECK();
    }
    
    /* create an instance of this object that has properly
       initialized devices */
    DeviceGroup *DeviceGroup::create(const int *deviceIDs,
                                     size_t     numDevices)
    {
      std::vector<int> tmpDeviceIDs;
      if (deviceIDs == 0) {
        for (int i=0;i<numDevices;i++)
          tmpDeviceIDs.push_back(i);
        deviceIDs = tmpDeviceIDs.data();
      }
      
      assert((deviceIDs == nullptr && numDevices == 0)
             ||
             (deviceIDs != nullptr && numDevices > 0)
             );
      
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
      LOG("initializing optix 7");
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
          std::cout << OWL_TERMINAL_RED
                    << "#owl.ll: Error creating optix device on CUDA device #"
                    << deviceIDs[i] << ": " << e.what() << " ... dropping this device"
                    << OWL_TERMINAL_DEFAULT << std::endl;
        }
      }

      // ------------------------------------------------------------------
      // some final sanity check that we managed to create at least
      // one device...
      // ------------------------------------------------------------------
      if (devices.empty())
        throw std::runtime_error("fatal error - could not find/create any optix devices");

      LOG_OK("successfully created device group with " << devices.size() << " devices");
      return new DeviceGroup(devices);
    }

  } // ::owl::ll
} //::owl

