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
#include "ll/DeviceMemory.h"

namespace owl {
  namespace ll {
    struct ProgramGroups;
    struct Device;
    
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
    
    struct Context {
      
      Context(int owlDeviceID, int cudaDeviceID);
      ~Context();
      
      /*! linear ID (0,1,2,...) of how *we* number devices (ie,
        'first' device is alwasys device 0, no matter if it runs on
        another physical/cuda device */
      const int          owlDeviceID;
      
      /* the cuda device ID that this logical device runs on */
      const int          cudaDeviceID;

      void setActive() { CUDA_CHECK(cudaSetDevice(cudaDeviceID)); }
      void pushActive()
      {
        assert("check we're not already pushed" && savedActiveDeviceID == -1);
        CUDA_CHECK(cudaGetDevice(&savedActiveDeviceID));
        setActive();
      }
      void popActive()
      {
        assert("check we do have a saved device" && savedActiveDeviceID >= 0);
        CUDA_CHECK(cudaSetDevice(savedActiveDeviceID));
        savedActiveDeviceID = -1;
      }
      int  savedActiveDeviceID = -1;
      
      void createPipeline(Device *device);
      void destroyPipeline();
      
      OptixDeviceContext optixContext = nullptr;
      CUcontext          cudaContext  = nullptr;
      CUstream           stream       = nullptr;

      OptixPipelineCompileOptions pipelineCompileOptions = {};
      OptixPipelineLinkOptions    pipelineLinkOptions    = {};
      OptixModuleCompileOptions   moduleCompileOptions   = {};
      OptixPipeline               pipeline               = nullptr;

      int numRayTypes { 1 };
    };
    
    struct Module {
      OptixModule module = nullptr;
      const char *ptxCode;
      void create(Context *context);
    };
    struct Modules {
      ~Modules() {
        assert(noActiveHandles());
      }
      bool noActiveHandles() {
        for (auto &module : modules) if (module.module != nullptr) return false;
        return true;
      }
      size_t size() const { return modules.size(); }
      void alloc(size_t size);
      /*! will destroy the *optix handles*, but will *not* clear the
        modules vector itself */
      void destroyOptixHandles(Context *context);
      void buildOptixHandles(Context *context);
      void set(size_t slot, const char *ptxCode);
      Module *get(int moduleID) { return moduleID < 0?nullptr:&modules[moduleID]; }
      std::vector<Module> modules;
    };
    
    struct ProgramGroup {
      // OptixProgramGroupOptions  pgOptions = {};
      // OptixProgramGroupDesc     pgDesc;
      OptixProgramGroup         pg        = nullptr;
    };
    struct Program {
      const char *progName = nullptr;
      int         moduleID = -1;
    };
    struct RayGenPG : public ProgramGroup {
      Program program;
    };
    struct MissPG : public ProgramGroup {
      Program program;
    };
    struct HitGroupPG : public ProgramGroup {
      Program anyHit;
      Program closestHit;
      Program intersect;
    };
    // struct ProgramGroups {
    //   std::vector<HitGroupPG> hitGroupPGs;
    //   std::vector<RayGenPG> rayGenPGs;
    //   std::vector<MissPG> missPGs;
    // };

    struct SBT {
      OptixShaderBindingTable sbt = {};
      DeviceMemory raygenRecordsBuffer;
      DeviceMemory missRecordsBuffer;
      DeviceMemory hitGroupRecordsBuffer;
      DeviceMemory launchParamBuffer;
    };

    typedef enum { TRIANGLES, USER } GeomType;
    
    // struct StridedDeviceData
    // {
    //   size_t stride    = 0;
    //   size_t offset    = 0;
    //   size_t count     = 0;
    //   void  *d_pointer = 0;
    // };

    struct Buffer : public DeviceMemory {
      Buffer(const size_t elementCount,
             const size_t elementSize)
        : elementCount(elementCount),
          elementSize(elementSize)
      {
        assert(elementSize > 0);
        alloc(elementCount*elementSize);
      }
      const size_t elementCount;
      const size_t elementSize;
      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geom/group that is still being refrerenced by a
          group. Note we wil *NOT* automatically free a buffer if
          refcount reaches zero - this is ONLY for sanity checking
          during object deletion */
      int numTimesReferenced = 0;
    };
    
    struct Geom {
      Geom(int logicalHitGroupID)
        : logicalHitGroupID(logicalHitGroupID)
      {}
      virtual GeomType type() = 0;
      
      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geom/group that is still being refrerenced by a
          group. Note we wil *NOT* automatically free a buffer if
          refcount reaches zero - this is ONLY for sanity checking
          during object deletion */
      int numTimesReferenced = 0;
      const int logicalHitGroupID;
    };
    struct UserGeom : public Geom {
      virtual GeomType type() { return USER; }
      
      DeviceMemory bounds;
    };
    struct TrianglesGeom : public Geom {
      TrianglesGeom(int logicalHitGroupID)
        : Geom(logicalHitGroupID)
      {}
      virtual GeomType type() { return TRIANGLES; }

      void  *vertexPointer = nullptr;
      size_t vertexStride  = 0;
      size_t vertexCount   = 0;
      void  *indexPointer  = nullptr;
      size_t indexStride   = 0;
      size_t indexCount    = 0;
    };
    
    struct Group {
      virtual bool containsGeom() = 0;
      inline  bool containsInstances() { return !containsGeom(); }

      virtual void destroyAccel(Context *context) = 0;
      virtual void buildAccel(Context *context) = 0;
      
      std::vector<int>       elements;
      OptixTraversableHandle traversable = 0;
      DeviceMemory           bvhMemory;

      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geom/group that is still being refrerenced by a
          group. */
      int numTimesReferenced = 0;
    };
    struct InstanceGroup : public Group {
      virtual bool containsGeom() { return false; }
      
      virtual void destroyAccel(Context *context) override
      { OWL_NOTIMPLEMENTED; }
      virtual void buildAccel(Context *context) override
      { OWL_NOTIMPLEMENTED; }
    };
    struct GeomGroup : public Group {
      GeomGroup(size_t numChildren)
        : children(numChildren)
      {}
      
      virtual bool containsGeom() { return true; }
      virtual GeomType geomType() = 0;

      std::vector<Geom *> children;
    };
    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(size_t numChildren)
        : GeomGroup(numChildren)
      {}
      virtual GeomType geomType() { return TRIANGLES; }
      
      virtual void destroyAccel(Context *context) override;
      virtual void buildAccel(Context *context) override;
    };
    struct UserGeomGroup : public GeomGroup {
      virtual GeomType geomType() { return USER; }
      
      virtual void destroyAccel(Context *context) override
      { OWL_NOTIMPLEMENTED; }
      virtual void buildAccel(Context *context) override
      { OWL_NOTIMPLEMENTED; }
    };


    struct Device {

      /*! construct a new owl device on given cuda device. throws an
          exception if for any reason that cannot be done */
      Device(int owlDeviceID, int cudaDeviceID);
      ~Device();

      void createPipeline()
      {
        context->createPipeline(this);
      }
      
      void destroyPipeline()
      {
        context->destroyPipeline();
      }

      void buildModules()
      {
        // modules shouldn't be rebuilt while a pipeline is still using them(?)
        assert(context->pipeline == nullptr);
        modules.destroyOptixHandles(context);
        modules.buildOptixHandles(context);
      }

      /*! (re-)builds all optix programs, with current pipeline settings */
      void buildPrograms()
      {
        // programs shouldn't be rebuilt while a pipeline is still using them(?)
        assert(context->pipeline == nullptr);
        destroyOptixPrograms();
        buildOptixPrograms();
      }

      void setHitGroupClosestHit(int pgID, int moduleID, const char *progName);
      void setRayGenPG(int pgID, int moduleID, const char *progName);
      void setMissPG(int pgID, int moduleID, const char *progName);
      
      void allocModules(size_t count)
      { modules.alloc(count); }
      /*! each geom will always use "numRayTypes" successive hit
          groups (one per ray type), so this must be a multiple of the
          number of ray types to be used */
      void allocHitGroupPGs(size_t count);
      void allocRayGenPGs(size_t count);
      /*! each geom will always use "numRayTypes" successive hit
          groups (one per ray type), so this must be a multiple of the
          number of ray types to be used */
      void allocMissPGs(size_t count);

      /*! resize the array of geom IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geoms that would get 'lost' have alreay been
          destroyed */
      void reallocGeoms(size_t newCount)
      {
        for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)geoms.size();idxWeWouldLose++)
          assert("realloc would lose a geom that was not properly destroyed" &&
                 geoms[idxWeWouldLose] == nullptr);
        geoms.resize(newCount);
      }


      void createTrianglesGeom(int geomID,
                                   /*! the "logical" hit group ID:
                                       will always count 0,1,2... evne
                                       if we are using multiple ray
                                       types; the actual hit group
                                       used when building the SBT will
                                       then be 'logicalHitGroupID *
                                       numRayTypes) */
                                   int logicalHitGroupID)
      {
        assert("check ID is valid" && geomID >= 0);
        assert("check ID is valid" && geomID < geoms.size());
        assert("check given ID isn't still in use" && geoms[geomID] == nullptr);

        assert("check valid hit group ID" && logicalHitGroupID >= 0);
        assert("check valid hit group ID"
               && logicalHitGroupID*context->numRayTypes < hitGroupPGs.size());
        
        geoms[geomID] = new TrianglesGeom(logicalHitGroupID);
        assert("check 'new' was successful" && geoms[geomID] != nullptr);
      }

      /*! resize the array of geom IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geoms that would get 'lost' have alreay been
          destroyed */
      void reallocGroups(size_t newCount)
      {
        for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)groups.size();idxWeWouldLose++)
          assert("realloc would lose a geom that was not properly destroyed" &&
                 groups[idxWeWouldLose] == nullptr);
        groups.resize(newCount);
      }

      /*! resize the array of buffer handles. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          buffer handles that would get 'lost' have alreay been
          destroyed */
      void reallocBuffers(size_t newCount)
      {
        for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)buffers.size();idxWeWouldLose++)
          assert("realloc would lose a geom that was not properly destroyed" &&
                 buffers[idxWeWouldLose] == nullptr);
        buffers.resize(newCount);
      }
      
      void createTrianglesGeomGroup(int groupID,
                                        int *geomIDs,
                                        int geomCount)
      {
        assert("check for valid ID" && groupID >= 0);
        assert("check for valid ID" && groupID < groups.size());
        assert("check group ID is available" && groups[groupID] ==nullptr);
        
        assert("check for valid combinations of child list" &&
               ((geomIDs == nullptr && geomCount == 0) ||
                (geomIDs != nullptr && geomCount >  0)));
        
        TrianglesGeomGroup *group = new TrianglesGeomGroup(geomCount);
        assert("check 'new' was successful" && group != nullptr);
        groups[groupID] = group;

        // set children - todo: move to separate (api?) function(s)!?
        for (int childID=0;childID<geomCount;childID++) {
          int geomID = geomIDs[childID];
          assert("check geom child geom ID is valid" && geomID >= 0);
          assert("check geom child geom ID is valid" && geomID <  geoms.size());
          Geom *geom = geoms[geomID];
          assert("check geom indexed child geom valid" && geom != nullptr);
          assert("check geom is valid type" && geom->type() == TRIANGLES);
          geom->numTimesReferenced++;
          group->children[childID] = geom;
        }
      }

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
      
      void destroyGeom(size_t ID)
      {
        assert("check for valid ID" && ID >= 0);
        assert("check for valid ID" && ID < geoms.size());
        assert("check still valid"  && geoms[ID] != nullptr);
        // set to null, which should automatically destroy
        geoms[ID] = nullptr;
      }

      /*! for each valid program group, use optix to compile/build the
          acutal program to an optix-usable form (ie, this builds the
          OptixProgramGroup object for each PG) */
      void buildOptixPrograms();

      /*! destroys all currently active OptixProgramGroup group
        objects in all our PG vectors, but NOT those PG vectors
        themselves; ie, we can always call 'buildOptixPrograms' to
        re-build them (which allows, for example, to
        'destroyOptixPrograms', chance compile/link options, and
        rebuild them) */
      void destroyOptixPrograms();

      // ------------------------------------------------------------------
      // group related struff
      // ------------------------------------------------------------------
      void groupBuildAccel(int groupID);

      // accessor helpers:
      Geom *checkGetGeom(int geomID)
      {
        assert("check valid geom ID" && geomID >= 0);
        assert("check valid geom ID" && geomID <  geoms.size());
        Geom *geom = geoms[geomID];
        assert("check valid geom" && geom != nullptr);
        return geom;
      }

      // accessor helpers:
      Group *checkGetGroup(int groupID)
      {
        assert("check valid group ID" && groupID >= 0);
        assert("check valid group ID" && groupID <  groups.size());
        Group *group = groups[groupID];
        assert("check valid group" && group != nullptr);
        return group;
      }

      Buffer *checkGetBuffer(int bufferID)
      {
        assert("check valid geom ID" && bufferID >= 0);
        assert("check valid geom ID" && bufferID <  buffers.size());
        Buffer *buffer = buffers[bufferID];
        assert("check valid buffer" && buffer != nullptr);
        return buffer;
      }

      // accessor helpers:
      TrianglesGeom *checkGetTrianglesGeom(int geomID)
      {
        Geom *geom = checkGetGeom(geomID);
        assert(geom);
        TrianglesGeom *asTriangles
          = dynamic_cast<TrianglesGeom*>(geom);
        assert("check geom is triangle geom" && asTriangles != nullptr);
        return asTriangles;
      }

      void sbtHitGroupsBuild(size_t maxHitGroupDataSize,
                             WriteHitGroupCallBack writeHitGroupCallBack,
                             const void *callBackUserData);
      
      Context                  *context;
      
      Modules                   modules;
      std::vector<HitGroupPG>   hitGroupPGs;
      std::vector<RayGenPG>     rayGenPGs;
      std::vector<MissPG>       missPGs;
      std::vector<Geom *>   geoms;
      std::vector<Group *>      groups;
      std::vector<Buffer *>     buffers;
      SBT                       sbt;
    };
    
    struct DeviceGroup {
      typedef std::shared_ptr<DeviceGroup> SP;

      DeviceGroup(const std::vector<Device *> &devices);
      DeviceGroup()
      {
        std::cout << "#owl.ll: destroying devices" << std::endl;
        for (auto device : devices) {
          assert(device);
          delete device;
        }
        std::cout << GDT_TERMINAL_GREEN
                  << "#owl.ll: all devices properly destroyed"
                  << GDT_TERMINAL_DEFAULT << std::endl;
      }
      
      void allocModules(size_t count)
      { for (auto device : devices) device->allocModules(count); }
      void setModule(size_t slot, const char *ptxCode)
      { for (auto device : devices) device->modules.set(slot,ptxCode); }
      void buildModules()
      {
        for (auto device : devices)
          device->buildModules();
      }
      void buildPrograms()
      {
        for (auto device : devices)
          device->buildPrograms();
      }
      void createPipeline()
      { for (auto device : devices) device->createPipeline(); }
      
      void allocHitGroupPGs(size_t count)
      { for (auto device : devices) device->allocHitGroupPGs(count); }
      void allocRayGenPGs(size_t count)
      { for (auto device : devices) device->allocRayGenPGs(count); }
      void allocMissPGs(size_t count)
      { for (auto device : devices) device->allocMissPGs(count); }

      void setHitGroupClosestHit(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setHitGroupClosestHit(pgID,moduleID,progName); }
      void setRayGenPG(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setRayGenPG(pgID,moduleID,progName); }
      void setMissPG(int pgID, int moduleID, const char *progName)
      { for (auto device : devices) device->setMissPG(pgID,moduleID,progName); }
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void reallocGroups(size_t newCount)
      { for (auto device : devices) device->reallocGroups(newCount); }
      
      void reallocBuffers(size_t newCount)
      { for (auto device : devices) device->reallocBuffers(newCount); }
      
      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void reallocGeoms(size_t newCount)
      { for (auto device : devices) device->reallocGeoms(newCount); }

      void createTrianglesGeom(int geomID,
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
      
      void createTrianglesGeomGroup(int groupID,
                                        int *geomIDs, int geomCount)
      {
        assert("check for valid combinations of child list" &&
               ((geomIDs == nullptr && geomCount == 0) ||
                (geomIDs != nullptr && geomCount >  0)));
        
        for (auto device : devices) {
          device->createTrianglesGeomGroup(groupID,geomIDs,geomCount);
        }
      }

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
      void groupBuildAccel(int groupID)
      {
        for (auto device : devices) 
          device->groupBuildAccel(groupID);
      }
      
      void sbtHitGroupsBuild(size_t maxHitGroupDataSize,
                             WriteHitGroupCallBack writeHitGroupCallBack,
                             void *callBackData)
      {
        for (auto device : devices) 
          device->sbtHitGroupsBuild(maxHitGroupDataSize,
                                    writeHitGroupCallBack,
                                    callBackData);
      }

      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs. Note this is
         the only shared_ptr we use on that abstractoin level, but
         here we use one to force a proper destruction of the
         device */
      static DeviceGroup::SP create(const int *deviceIDs  = nullptr,
                                    size_t     numDevices = 0);

      const std::vector<Device *> devices;
    };
    
  } // ::owl::ll
} //::owl
