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
      /*! for all *optix* programs we can directly buidl the PTX code
          into a module using optixbuildmodule - this is the result of
          that operation */
      OptixModule module = nullptr;
      
      /*! for the *bounds* function we have to build a *separate*
          module because this one is built outside of optix, and thus
          does not have the internal _optix_xyz() symbols in it */
      CUmodule    boundsModule = 0;
      
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
      size_t      dataSize = 0;
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
      size_t  boundsProgDataSize = 0; // do we still need this!?
      CUfunction boundsFuncKernel;
    };

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
      Geom(int geomID, int geomTypeID)
        : geomID(geomID), geomTypeID(geomTypeID)
      {}
      virtual PrimType primType() = 0;
      
      /*! only for error checking - we do NOT do reference counting
        ourselves, but will use this to track erorrs like destroying
        a geom/group that is still being refrerenced by a
        group. Note we wil *NOT* automatically free a buffer if
        refcount reaches zero - this is ONLY for sanity checking
        during object deletion */
      int numTimesReferenced = 0;
      const int geomID;
      const int geomTypeID;
    };
    struct UserGeom : public Geom {
      UserGeom(int geomID, int geomTypeID, int numPrims)
        : Geom(geomID,geomTypeID),
          numPrims(numPrims)
      {}
      virtual PrimType primType() { return USER; }

      /*! the pointer to the device-side bounds array. Note this
          pointer _can_ be the same as 'boundsBuffer' (if *we* manage
          that memory), but in the case of user-supplied bounds buffer
          it is also possible that boundsBuffer is not allocated, and
          d_boundsArray points to the user-supplied buffer */
      void        *d_boundsMemory = nullptr;
      DeviceMemory internalBufferForBoundsProgram;
      size_t       numPrims      = 0;
    };
    struct TrianglesGeom : public Geom {
      TrianglesGeom(int geomID, int geomTypeID)
        : Geom(geomID, geomTypeID)
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
      
      /*! set closest hit program for given geometry type and ray
          type. Note progName will *not* be copied, so the pointer
          must remain valid as long as this geom may ever get
          recompiled */
      void setGeomTypeClosestHit(int geomTypeID,
                                 int rayTypeID,
                                 int moduleID,
                                 const char *progName);
      
      /*! set intersect program for given geometry type and ray type
        (only allowed for user geometry types). Note progName will
        *not* be copied, so the pointer must remain valid as long as
        this geom may ever get recompiled */
      void setGeomTypeIntersect(int geomTypeID,
                                int rayTypeID,
                                int moduleID,
                                const char *progName);
      
      void setRayGen(int pgID,
                     int moduleID,
                     const char *progName,
                     /*! size of that program's SBT data */
                     size_t missProgDataSize);
      
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
                       /*! size of that program's SBT data */
                       size_t missProgDataSize);
      
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

      GeomType *checkGetGeomType(int geomTypeID)
      {
        assert("check valid geomType ID" && geomTypeID >= 0);
        assert("check valid geomType ID" && geomTypeID <  geomTypes.size());
        GeomType *geomType = &geomTypes[geomTypeID];
        assert("check valid geomType" && geomType != nullptr);
        return geomType;
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

      // accessor helpers:
      UserGeomGroup *checkGetUserGeomGroup(int groupID)
      {
        assert("check valid group ID" && groupID >= 0);
        assert("check valid group ID" && groupID <  groups.size());
        Group *group = groups[groupID];
        assert("check valid group" && group != nullptr);
        UserGeomGroup *ugg = dynamic_cast<UserGeomGroup*>(group);
        if (!ugg)
          OWL_EXCEPT("group is not a user geometry group");
        assert("check group is a user geom group" && ugg != nullptr);
        return ugg;
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

      // accessor helpers:
      UserGeom *checkGetUserGeom(int geomID)
      {
        Geom *geom = checkGetGeom(geomID);
        assert(geom);
        UserGeom *asUser
          = dynamic_cast<UserGeom*>(geom);
        assert("check geom is triangle geom" && asUser != nullptr);
        return asUser;
      }

      /*! only valid for a user geometry group - (re-)builds the
          primitive bounds array required for building the
          acceleration structure by executing the device-side bounding
          box program */
      void groupBuildPrimitiveBounds(int groupID,
                                     size_t maxGeomDataSize,
                                     WriteUserGeomBoundsDataCB cb,
                                     void *cbData);
      void sbtGeomTypesBuild(size_t maxHitProgDataSize,
                             WriteHitProgDataCB writeHitProgDataCB,
                             const void *callBackUserData);
      void sbtRayGensBuild(WriteRayGenDataCB writeRayGenDataCB,
                           const void *callBackUserData);
      void sbtMissProgsBuild(WriteMissProgDataCB writeMissProgDataCB,
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
