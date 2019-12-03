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
// for the hit group callback type, which is part of the API
#include "ll/DeviceGroup.h"

namespace owl {
  namespace ll {
    struct ProgramGroups;
    struct Device;
    
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
    struct MissProgPG : public ProgramGroup {
      Program program;
    };
    struct HitGroupPG : public ProgramGroup {
      Program anyHit;
      Program closestHit;
      Program intersect;
    };
    struct GeomType {
      std::vector<HitGroupPG> perRayType;
      Program boundsProg;
    };
    // struct ProgramGroups {
    //   std::vector<HitGroupPG> hitGroupPGs;
    //   std::vector<RayGenPG> rayGenPGs;
    //   std::vector<MissPG> missProgPGs;
    // };

    struct SBT {
      OptixShaderBindingTable sbt = {};
      
      size_t rayGenRecordCount   = 0;
      size_t rayGenRecordSize   = 0;
      DeviceMemory rayGenRecordsBuffer;

      size_t hitGroupRecordSize = 0;
      size_t hitGroupRecordCount = 0;
      DeviceMemory hitGroupRecordsBuffer;

      size_t missProgRecordSize = 0;
      size_t missProgRecordCount = 0;
      DeviceMemory missProgRecordsBuffer;

      DeviceMemory launchParamsBuffer;
    };

    typedef enum { TRIANGLES, USER } PrimType;
    
    struct Buffer {
      Buffer(const size_t elementCount,
             const size_t elementSize)
        : elementCount(elementCount),
          elementSize(elementSize)
      {
        assert(elementSize > 0);
      }
      virtual ~Buffer()
      {
        assert(numTimesReferenced == 0);
      }
      inline void *get() const { return d_pointer; }
      const size_t elementCount;
      const size_t elementSize;
      void        *d_pointer = nullptr;
      /*! only for error checking - we do NOT do reference counting
        ourselves, but will use this to track erorrs like destroying
        a geom/group that is still being refrerenced by a
        group. Note we wil *NOT* automatically free a buffer if
        refcount reaches zero - this is ONLY for sanity checking
        during object deletion */
      int numTimesReferenced = 0;
    };

    struct DeviceBuffer : public Buffer
    {
      DeviceBuffer(const size_t elementCount,
                   const size_t elementSize)
        : Buffer(elementCount, elementSize)
      {
        devMem.alloc(elementCount*elementSize);
        d_pointer = devMem.get();
      }
      ~DeviceBuffer()
      {
        devMem.free();
      }
      DeviceMemory devMem;
    };
    
    struct HostPinnedBuffer : public Buffer
    {
      HostPinnedBuffer(const size_t elementCount,
                       const size_t elementSize,
                       HostPinnedMemory::SP pinnedMem)
        : Buffer(elementCount, elementSize),
          pinnedMem(pinnedMem)
      {
        d_pointer = pinnedMem->pointer;
      }
      HostPinnedMemory::SP pinnedMem;
    };
      
    struct Geom {
      Geom(int geomTypeID)
        : geomTypeID(geomTypeID)
      {}
      virtual PrimType primType() = 0;
      
      /*! only for error checking - we do NOT do reference counting
        ourselves, but will use this to track erorrs like destroying
        a geom/group that is still being refrerenced by a
        group. Note we wil *NOT* automatically free a buffer if
        refcount reaches zero - this is ONLY for sanity checking
        during object deletion */
      int numTimesReferenced = 0;
      const int geomTypeID;
    };
    struct UserGeom : public Geom {
      UserGeom(int geomTypeID, int numPrims)
        : Geom(geomTypeID),
          numPrims(numPrims)
      {}
      virtual PrimType primType() { return USER; }
      
      DeviceMemory boundsBuffer;
      size_t       numPrims;
    };
    struct TrianglesGeom : public Geom {
      TrianglesGeom(int geomTypeID)
        : Geom(geomTypeID)
      {}
      virtual PrimType primType() { return TRIANGLES; }

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
      virtual PrimType primType() = 0;

      std::vector<Geom *> children;
    };
    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(size_t numChildren)
        : GeomGroup(numChildren)
      {}
      virtual PrimType primType() { return TRIANGLES; }
      
      virtual void destroyAccel(Context *context) override;
      virtual void buildAccel(Context *context) override;
    };
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(size_t numChildren)
        : GeomGroup(numChildren)
      {}
      virtual PrimType primType() { return USER; }
      
      virtual void destroyAccel(Context *context) override;
      virtual void buildAccel(Context *context) override;
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

      void setGeomTypeClosestHit(int pgID, int rayTypeID, int moduleID, const char *progName);
      void setRayGen(int pgID, int moduleID, const char *progName);
      void setMissProg(int pgID, int moduleID, const char *progName);
      
      void allocModules(size_t count)
      { modules.alloc(count); }
      /*! each geom will always use "numRayTypes" successive hit
        groups (one per ray type), so this must be a multiple of the
        number of ray types to be used */
      void allocGeomTypes(size_t count);

      void allocRayGens(size_t count);
      /*! each geom will always use "numRayTypes" successive hit
        groups (one per ray type), so this must be a multiple of the
        number of ray types to be used */
      void allocMissProgs(size_t count);

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

      void createUserGeom(int geomID,
                          /*! the "logical" hit group ID:
                            will always count 0,1,2... evne
                            if we are using multiple ray
                            types; the actual hit group
                            used when building the SBT will
                            then be 'geomTypeID *
                            numRayTypes) */
                          int geomTypeID,
                          int numPrims);

      void createTrianglesGeom(int geomID,
                               /*! the "logical" hit group ID:
                                 will always count 0,1,2... evne
                                 if we are using multiple ray
                                 types; the actual hit group
                                 used when building the SBT will
                                 then be 'geomTypeID *
                                 numRayTypes) */
                               int geomTypeID);

      /*! resize the array of geom IDs. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed */
      void reallocGroups(size_t newCount);

      /*! resize the array of buffer handles. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        buffer handles that would get 'lost' have alreay been
        destroyed */
      void reallocBuffers(size_t newCount);
      
      void createTrianglesGeomGroup(int groupID,
                                    int *geomIDs,
                                    int geomCount);

      void createUserGeomGroup(int groupID,
                               int *geomIDs,
                               int geomCount);

      /*! returns the given buffers device pointer */
      void *bufferGetPointer(int bufferID);
      void createDeviceBuffer(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);
      void createHostPinnedBuffer(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize,
                                  HostPinnedMemory::SP pinnedMem);
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
      
      /*! return given group's current traversable. note this function
        will *not* check if the group has alreadybeen built, if it
        has to be rebuilt, etc. */
      OptixTraversableHandle groupGetTraversable(int groupID);
      
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

      void sbtGeomTypesBuild(size_t maxHitProgDataSize,
                             WriteHitProgDataCB writeHitProgDataCB,
                             const void *callBackUserData);
      void sbtRayGensBuild(size_t maxRayGenDataSize,
                           WriteRayGenDataCB writeRayGenDataCB,
                           const void *callBackUserData);
      void sbtMissProgsBuild(size_t maxMissProgDataSize,
                             WriteMissProgDataCB writeMissProgDataCB,
                             const void *callBackUserData);

      void launch(int rgID, const vec2i &dims);
      
      Context                  *context;
      
      Modules                   modules;
      std::vector<GeomType>     geomTypes;
      std::vector<RayGenPG>     rayGenPGs;
      std::vector<MissProgPG>   missProgPGs;
      std::vector<Geom *>       geoms;
      std::vector<Group *>      groups;
      std::vector<Buffer *>     buffers;
      SBT                       sbt;
    };
    
  } // ::owl::ll
} //::owl
