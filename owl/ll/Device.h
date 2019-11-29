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

#include "abstract/common.h"
#include "ll/optix.h"
#include "ll/DeviceMemory.h"

namespace owl {
  namespace ll {
    struct ProgramGroups;
    struct Device;
    
    struct Context {
      typedef std::shared_ptr<Context> SP;
      
      Context(int owlDeviceID, int cudaDeviceID);
      ~Context();
      
      /*! linear ID (0,1,2,...) of how *we* number devices (ie,
        'first' device is alwasys device 0, no matter if it runs on
        another physical/cuda device */
      const int          owlDeviceID;
      
      /* the cuda device ID that this logical device runs on */
      const int          cudaDeviceID;

      void setActive() { CUDA_CHECK(cudaSetDevice(cudaDeviceID)); }

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

    typedef enum { TRIANGLES, USER } GeometryType;
    struct StridedBuffer : public DeviceMemory {
      size_t stride = 0;
      size_t offset = 0;
      size_t count  = 0;
      bool   memoryIsOwnedByUs = false;
    };

    struct Buffer {
      Buffer(const size_t elementCount,
             const size_t elementSize)
        : elementCount(elementCount),
          elementSize(elementSize)
      {}
      const size_t elementCount;
      const size_t elementSize;
      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geometry/group that is still being refrerenced by a
          group. Note we wil *NOT* automatically free a buffer if
          refcount reaches zero - this is ONLY for sanity checking
          during object deletion */
      int numTimesReferenced = 0;
    };
    
    struct Geometry {
      Geometry(int logicalHitGroupID)
        : logicalHitGroupID(logicalHitGroupID)
      {}
      virtual GeometryType type() = 0;
      
      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geometry/group that is still being refrerenced by a
          group. Note we wil *NOT* automatically free a buffer if
          refcount reaches zero - this is ONLY for sanity checking
          during object deletion */
      int numTimesReferenced = 0;
      const int logicalHitGroupID;
    };
    struct UserGeometry : public Geometry {
      virtual GeometryType type() { return USER; }
      
      StridedBuffer bounds;
    };
    struct TrianglesGeometry : public Geometry {
      TrianglesGeometry(int logicalHitGroupID)
        : Geometry(logicalHitGroupID)
      {}
      virtual GeometryType type() { return TRIANGLES; }

      StridedBuffer vertices;
      StridedBuffer indices;
      
      DeviceMemory indicesBuffer;
      size_t       indicesStride;
      size_t       indicesCount;
      //! whether it was *us* that alloc'ed the indices array, or somebody else
      bool         indicesAreOurs;
    };
    
    struct Group {
      virtual bool containsGeometry() = 0;
      inline  bool containsInstances() { return !containsGeometry(); }
      
      std::vector<int>       elements;
      OptixTraversableHandle traversable;
      DeviceMemory           bvhMemory;

      /*! only for error checking - we do NOT do reference counting
          ourselves, but will use this to track erorrs like destroying
          a geometry/group that is still being refrerenced by a
          group. */
      int numTimesReferenced = 0;
    };
    struct InstanceGroup : public Group {
      virtual bool containsGeometry() { return false; }
    };
    struct GeometryGroup : public Group {
      GeometryGroup(size_t numChildren)
        : children(numChildren)
      {}
      
      virtual bool containsGeometry() { return true; }
      virtual GeometryType geometryType() = 0;

      std::vector<Geometry *> children;
    };
    struct TrianglesGeometryGroup : public GeometryGroup {
      TrianglesGeometryGroup(size_t numChildren)
        : GeometryGroup(numChildren)
      {}
      virtual GeometryType geometryType() { return TRIANGLES; }
    };
    struct UserGeometryGroup : public GeometryGroup {
      virtual GeometryType geometryType() { return USER; }
    };


    struct Device {
      typedef std::shared_ptr<Device> SP;

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
        modules.destroyOptixHandles(context.get());
        modules.buildOptixHandles(context.get());
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
      /*! each geometry will always use "numRayTypes" successive hit
          groups (one per ray type), so this must be a multiple of the
          number of ray types to be used */
      void allocHitGroupPGs(size_t count);
      void allocRayGenPGs(size_t count);
      /*! each geometry will always use "numRayTypes" successive hit
          groups (one per ray type), so this must be a multiple of the
          number of ray types to be used */
      void allocMissPGs(size_t count);

      /*! resize the array of geometry IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geometries that would get 'lost' have alreay been
          destroyed */
      void reallocGeometries(size_t newCount)
      {
        for (int idxWeWouldLose=newCount;idxWeWouldLose<geometries.size();idxWeWouldLose++)
          assert("realloc would lose a geometry that was not properly destroyed" &&
                 geometries[idxWeWouldLose] == nullptr);
        geometries.resize(newCount);
      }


      void createTrianglesGeometry(int geomID,
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
        assert("check ID is valid" && geomID < geometries.size());
        assert("check given ID isn't still in use" && geometries[geomID] == nullptr);

        assert("check valid hit group ID" && logicalHitGroupID >= 0);
        assert("check valid hit group ID"
               && logicalHitGroupID*context->numRayTypes < hitGroupPGs.size());
        
        geometries[geomID] = new TrianglesGeometry(logicalHitGroupID);
        assert("check 'new' was successful" && geometries[geomID] != nullptr);
      }

      /*! resize the array of geometry IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geometries that would get 'lost' have alreay been
          destroyed */
      void reallocGroups(size_t newCount)
      {
        for (int idxWeWouldLose=newCount;idxWeWouldLose<groups.size();idxWeWouldLose++)
          assert("realloc would lose a geometry that was not properly destroyed" &&
                 groups[idxWeWouldLose] == nullptr);
        groups.resize(newCount);
      }

      /*! resize the array of buffer handles. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          buffer handles that would get 'lost' have alreay been
          destroyed */
      void reallocBuffers(size_t newCount)
      {
        for (int idxWeWouldLose=newCount;idxWeWouldLose<buffers.size();idxWeWouldLose++)
          assert("realloc would lose a geometry that was not properly destroyed" &&
                 buffers[idxWeWouldLose] == nullptr);
        buffers.resize(newCount);
      }
      
      void createTrianglesGeometryGroup(int groupID,
                                        int *geomIDs,
                                        int geomCount)
      {
        assert("check for valid ID" && groupID >= 0);
        assert("check for valid ID" && groupID < groups.size());
        assert("check group ID is available" && groups[groupID] ==nullptr);
        
        assert("check for valid combinations of child list" &&
               ((geomIDs == nullptr && geomCount == 0) ||
                (geomIDs != nullptr && geomCount >  0)));
        
        TrianglesGeometryGroup *group = new TrianglesGeometryGroup(geomCount);
        assert("check 'new' was successful" && group != nullptr);
        groups[groupID] = group;

        // set children - todo: move to separate (api?) function(s)!?
        for (int childID=0;childID<geomCount;childID++) {
          int geomID = geomIDs[childID];
          assert("check geom child geom ID is valid" && geomID >= 0);
          assert("check geom child geom ID is valid" && geomID <  geometries.size());
          Geometry *geom = geometries[geomID];
          assert("check geom indexed child geom valid" && geom != nullptr);
          assert("check geom is valid type" && geom->type() == TRIANGLES);
          geom->numTimesReferenced++;
          group->children[childID] = geom;
        }
      }

      void destroyGeometry(size_t ID)
      {
        assert("check for valid ID" && ID >= 0);
        assert("check for valid ID" && ID < geometries.size());
        assert("check still valid"  && geometries[ID] != nullptr);
        // set to null, which should automatically destroy
        geometries[ID] = nullptr;
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
      
      Context::SP               context;
      
      Modules                   modules;
      std::vector<HitGroupPG>   hitGroupPGs;
      std::vector<RayGenPG>     rayGenPGs;
      std::vector<MissPG>       missPGs;
      std::vector<Geometry *>   geometries;
      std::vector<Group *>      groups;
      std::vector<Buffer *>     buffers;
      SBT                       sbt;
    };
    
    struct DeviceGroup {
      typedef std::shared_ptr<DeviceGroup> SP;

      DeviceGroup(const std::vector<Device::SP> &devices);

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
      
      /*! resize the array of geometry IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geometries that would get 'lost' have alreay been
          destroyed */
      void reallocGroups(size_t newCount)
      { for (auto device : devices) device->reallocGroups(newCount); }
      
      void reallocBuffers(size_t newCount)
      { for (auto device : devices) device->reallocBuffers(newCount); }
      
      /*! resize the array of geometry IDs. this can be either a
          'grow' or a 'shrink', but 'shrink' is only allowed if all
          geometries that would get 'lost' have alreay been
          destroyed */
      void reallocGeometries(size_t newCount)
      { for (auto device : devices) device->reallocGeometries(newCount); }

      void createTrianglesGeometry(int geomID,
                                   /*! the "logical" hit group ID:
                                       will always count 0,1,2... evne
                                       if we are using multiple ray
                                       types; the actual hit group
                                       used when building the SBT will
                                       then be 'logicalHitGroupID *
                                       numRayTypes) */
                                   int logicalHitGroupID)
      {
        for (auto device : devices)
          device->createTrianglesGeometry(geomID,logicalHitGroupID);
      }
      
      void createTrianglesGeometryGroup(int groupID,
                                        int *geomIDs, int geomCount)
      {
        assert("check for valid combinations of child list" &&
               ((geomIDs == nullptr && geomCount == 0) ||
                (geomIDs != nullptr && geomCount >  0)));
        
        for (auto device : devices) {
          device->createTrianglesGeometryGroup(groupID,geomIDs,geomCount);
        }
      }

      void createDeviceBuffer(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
      void triangleGeometrySetVertices(int geomID,
                                       int bufferID,
                                       int count,
                                       int stride,
                                       int offset);
      void triangleGeometrySetIndices(int geomID,
                                      int bufferID,
                                      int stride,
                                      int offset);
      
      /* create an instance of this object that has properly
         initialized devices for given cuda device IDs */
      static DeviceGroup::SP create(const int *deviceIDs  = nullptr,
                                    size_t     numDevices = 0);
      
      const std::vector<Device::SP> devices;
    };
    
  } // ::owl::ll
} //::owl
