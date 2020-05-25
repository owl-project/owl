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

#pragma once

#include "owl/ll/helper/optix.h"
#include "owl/ll/DeviceMemory.h"
// for the hit group callback type, which is part of the API
#include "owl/ll/DeviceGroup.h"
#include "owl/ll/Buffers.h"

namespace owl {
  namespace ll {
    struct ProgramGroups;
    struct Device;
    
    /*! allocator that allows for allocating ranges of STB indices as
      required for adding groups of geometries to the SBT */
    struct RangeAllocator {
      int alloc(size_t size);
      void release(size_t begin, size_t size);
      size_t maxAllocedID = 0;
    private:
      struct FreedRange {
        size_t begin;
        size_t size;
      };
      std::vector<FreedRange> freedRanges;
    };


    struct Context {

      inline static bool logging() { return DeviceGroup::logging(); }
      
      Context(int owlDeviceID, int cudaDeviceID);
      ~Context();
      
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

      void createPipeline(Device *device);
      void destroyPipeline();

      
      /*! linear ID (0,1,2,...) of how *we* number devices (i.e.,
        'first' device is always device 0, no matter if it runs on
        another physical/cuda device) */
      const int          owlDeviceID;
      
      /* the cuda device ID that this logical device runs on */
      const int          cudaDeviceID;

      int  savedActiveDeviceID = -1;

      OptixDeviceContext optixContext = nullptr;
      CUcontext          cudaContext  = nullptr;
      CUstream           stream       = nullptr;

      /*! sets the pipelineCompileOptions etc based on
          maxConfiguredInstanceDepth */
      void configurePipelineOptions();
      
      OptixPipelineCompileOptions pipelineCompileOptions = {};
      OptixPipelineLinkOptions    pipelineLinkOptions    = {};
      OptixModuleCompileOptions   moduleCompileOptions   = {};
      OptixPipeline               pipeline               = nullptr;

      /*! maximum depth instancing tree as specified by
          `setMaxInstancingDepth` */
      int maxInstancingDepth = 1;      
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
      OptixProgramGroup         pg        = nullptr;
    };
    struct Program {
      const char *progName = nullptr;
      int         moduleID = -1;
      size_t      dataSize = 0;
    };
    struct LaunchParams {
      LaunchParams(Context *context, size_t sizeOfData);
      
      const size_t         dataSize;
      
      /*! host-size memory for the launch paramters - we have a
          host-side copy, too, so we can leave the launch2D call
          without having to first wait for the cudaMemcpy to
          complete */
      std::vector<uint8_t> hostMemory;
      
      /*! the cuda device memory we copy the launch params to */
      DeviceMemory         deviceMemory;
      
      /*! a cuda stream we can use for the async upload and the
          following async launch */
      cudaStream_t         stream = nullptr;
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
      size_t  hitProgDataSize = (size_t)-1;
      CUfunction boundsFuncKernel;
    };
    
    
    struct SBT {
      size_t rayGenRecordCount   = 0;
      size_t rayGenRecordSize    = 0;
      DeviceMemory rayGenRecordsBuffer;

      size_t hitGroupRecordSize  = 0;
      size_t hitGroupRecordCount = 0;
      DeviceMemory hitGroupRecordsBuffer;

      size_t missProgRecordSize  = 0;
      size_t missProgRecordCount = 0;
      DeviceMemory missProgRecordsBuffer;

      DeviceMemory launchParamsBuffer;
      
      RangeAllocator rangeAllocator;
    };

    typedef enum { TRIANGLES, USER } PrimType;
    
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
      UserGeom(int geomID, int geomTypeID, size_t numPrims)
        : Geom(geomID,geomTypeID),
          numPrims(numPrims)
      {}
      virtual PrimType primType() { return USER; }
      void setPrimCount(size_t numPrims)
      {
        assert("check size hasn't previously been set (changing not yet implemented...)"
               && this->numPrims == 0);
        this->numPrims = numPrims;
      }
      
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
      virtual int  getSBTOffset() const = 0;
      
      // std::vector<int>       elements;
      OptixTraversableHandle traversable = 0;
      DeviceMemory           bvhMemory;

      /*! only for error checking - we do NOT do reference counting
        ourselves, but will use this to track erorrs like destroying
        a geom/group that is still being refrerenced by a
        group. */
      int numTimesReferenced = 0;
    };
    struct InstanceGroup : public Group {
      InstanceGroup(size_t numChildren)
        : children(numChildren)
      {}
      virtual bool containsGeom() { return false; }
      
      virtual void destroyAccel(Context *context) override;
      virtual void buildAccel(Context *context) override;
      virtual int  getSBTOffset() const override { return 0; }

      DeviceMemory optixInstanceBuffer;
      DeviceMemory outputBuffer;
      std::vector<Group *>  children;
      std::vector<uint32_t> instanceIDs;
      std::vector<affine3f> transforms;
    };

    /*! \warning currently using std::vector of *geoms*, but will have
        to eventually use geom *IDs* if we want(?) to allow side
        effects when changing geometries */
    struct GeomGroup : public Group {
      GeomGroup(size_t numChildren,
                size_t sbtOffset)
        : children(numChildren),
          sbtOffset(sbtOffset)
      {}
      
      virtual bool containsGeom() { return true; }
      virtual PrimType primType() = 0;
      virtual int  getSBTOffset() const override { return (int)sbtOffset; }

      std::vector<Geom *> children;
      const size_t sbtOffset;
    };
    struct TrianglesGeomGroup : public GeomGroup {
      TrianglesGeomGroup(size_t numChildren,
                         size_t sbtOffset)
        : GeomGroup(numChildren,
                    sbtOffset)
      {}
      virtual PrimType primType() { return TRIANGLES; }
      
      virtual void destroyAccel(Context *context) override;
      virtual void buildAccel(Context *context) override;
    };
    struct UserGeomGroup : public GeomGroup {
      UserGeomGroup(size_t numChildren,
                    size_t sbtOffset)
        : GeomGroup(numChildren,
                    sbtOffset)
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

      /*! helper function - return cuda name of this device */
      std::string getDeviceName() const;
      
      /*! helper function - return cuda device ID of this device */
      int getCudaDeviceID() const;
      
      /*! set the maximum instancing depth that will be allowed; '0'
          means 'no instancing, only bottom level accels', '1' means
          'only one single level of instances' (i.e., instancegroups
          never have children that are themselves instance groups),
          etc. Note we currently do *not* yet check the node graph as
          to whether it adheres to this value - if you use a node
          graph that's deeper than the value passed through this
          function you will most likely see optix crashing on you (and
          correctly so).

          Note this value will only take effect upon the next
          buildPrograms() and createPipeline(), so should be called
          *before* those functions get called */
      void setMaxInstancingDepth(int maxInstancingDepth);

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

      /*! Set bounding box program for given geometry type, using a
          bounding box program to be called on the device. Note that
          unlike other programs (intersect, closesthit, anyhit) these
          programs are not 'per ray type', but exist only once per
          geometry type. Obviously only allowed for user geometry
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
      
      /*! set any hit program for given geometry type and ray
          type. Note progName will *not* be copied, so the pointer
          must remain valid as long as this geom may ever get
          recompiled */
      void setGeomTypeAnyHit(int geomTypeID,
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

      void allocLaunchParams(size_t count);
      
      /*! each geom will always use "numRayTypes" successive hit
        groups (one per ray type), so this must be a multiple of the
        number of ray types to be used */
      void allocGeomTypes(size_t count);

      void geomTypeCreate(int geomTypeID,
                          size_t programDataSize);
      void launchParamsCreate(int launchParamsID,
                              size_t sizeOfData);
        
      void allocRayGens(size_t count);

      /*! each geom will always use "numRayTypes" successive hit
        groups (one per ray type), so this must be a multiple of the
        number of ray types to be used */
      void allocMissProgs(size_t count);

      /*! Resize the array of geom IDs. This can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        geoms that would get 'lost' have alreay been
        destroyed. */
      void allocGeoms(size_t newCount)
      {
        for (int idxWeWouldLose=(int)newCount;idxWeWouldLose<(int)geoms.size();idxWeWouldLose++)
          assert("alloc would lose a geom that was not properly destroyed" &&
                 geoms[idxWeWouldLose] == nullptr);
        geoms.resize(newCount);
      }

      void userGeomCreate(int geomID,
                          /*! the "logical" hit group ID:
                            will always count 0,1,2... evne
                            if we are using multiple ray
                            types; the actual hit group
                            used when building the SBT will
                            then be 'geomTypeID *
                            numRayTypes) */
                          int geomTypeID,
                          size_t numPrims);

      void userGeomSetPrimCount(int geomID,
                                size_t numPrims);

      void trianglesGeomCreate(int geomID,
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
      void allocGroups(size_t newCount);

      /*! resize the array of buffer handles. this can be either a
        'grow' or a 'shrink', but 'shrink' is only allowed if all
        buffer handles that would get 'lost' have alreay been
        destroyed */
      void allocBuffers(size_t newCount);

      void allocTextures(size_t newCount);

      void trianglesGeomGroupCreate(int groupID,
                                    const int *geomIDs,
                                    size_t geomCount);

      void userGeomGroupCreate(int groupID,
                               const int *geomIDs,
                               size_t geomCount);

      /*! create a new instance group with given list of children */
      void instanceGroupCreate(/*! the group we are defining */
                               int groupID,
                               size_t numChildren,
                               /* list of children. list can be
                                  omitted by passing a nullptr, but if
                                  not null this must be a list of
                                  'childCount' valid group ID */
                               const uint32_t *childGroupIDs,
                               const uint32_t *instIDs,
                               const affine3f *xfms);

      /*! set given child's instance transform. groupID must be a
          valid instance group, childID must be wihtin
          [0..numChildren) */
      void instanceGroupSetTransform(int groupID,
                                     int childNo,
                                     const affine3f &xfm);
      /*! set given child to {childGroupID+xfm}  */
      void instanceGroupSetChild(int groupID,
                                 int childNo,
                                 int childGroupID);
      void geomGroupSetChild(int groupID,
                             int childNo,
                             int childID);

      /*! destroy the given buffer, and release all host and/or device
          memory associated with it */
      void bufferDestroy(int bufferID);
      
      /*! returns the given buffers device pointer */
      void *bufferGetPointer(int bufferID);

      /*! return the cuda stream by the given launchparams object, on
        given device */
      cudaStream_t launchParamsGetStream(int lpID);
      
      void bufferResize(int bufferID, size_t newItemCount);
      void bufferUpload(int bufferID, const void *hostPtr);
      
      void deviceBufferCreate(int bufferID,
                              size_t elementCount,
                              size_t elementSize,
                              const void *initData);

      /*! create a managed memory buffer */
      void managedMemoryBufferCreate(int bufferID,
                                     size_t elementCount,
                                     size_t elementSize,
                                     ManagedMemory::SP managedMem);
      
      void hostPinnedBufferCreate(int bufferID,
                                  size_t elementCount,
                                  size_t elementSize,
                                  HostPinnedMemory::SP pinnedMem);

      void graphicsBufferCreate(int bufferID,
                                size_t elementCount,
                                size_t elementSize,
                                cudaGraphicsResource_t resource);

      void graphicsBufferMap(int bufferID);

      void graphicsBufferUnmap(int bufferID);

      /*! Set a buffer of bounding boxes that this user geometry will
          use when building the accel structure. This is one of
          multiple ways of specifying the bounding boxes for a user
          geometry (the other two being a) setting the geometry type's
          boundsFunc, or b) setting a host-callback fr computing the
          bounds). Only one of the three methods can be set at any
          given time. */
      void userGeomSetBoundsBuffer(int geomID, int bufferID);
      
      void trianglesGeomSetVertexBuffer(int geomID,
                                        int32_t bufferID,
                                        size_t count,
                                        size_t stride,
                                        size_t offset);
      void trianglesGeomSetIndexBuffer(int geomID,
                                       int32_t bufferID,
                                       size_t count,
                                       size_t stride,
                                       size_t offset);
      
      void destroyGeom(size_t ID)
      {
        assert("check for valid ID" && ID >= 0);
        assert("check for valid ID" && ID < geoms.size());
        assert("check still valid"  && geoms[ID] != nullptr);
        // set to null, which should automatically destroy
        geoms[ID] = nullptr;
      }

      /*! for each valid program group, use optix to compile/build the
        actual program to an optix-usable form (i.e., this builds the
        OptixProgramGroup object for each PG) */
      void buildOptixPrograms();

      /*! destroys all currently active OptixProgramGroup group
        objects in all our PG vectors, but NOT those PG vectors
        themselves; i.e., we can always call 'buildOptixPrograms' to
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
      uint32_t groupGetSBTOffset(int groupID);
      
      // accessor helpers:
      Geom *checkGetGeom(int geomID)
      {
        assert("check valid geom ID" && geomID >= 0);
        assert("check valid geom ID" && geomID <  geoms.size());
        Geom *geom = geoms[geomID];
        assert("check valid geom" && geom != nullptr);
        return geom;
      }
      LaunchParams *checkGetLaunchParams(int launchParamsID)
      {
        assert("check valid launchParams ID" && launchParamsID >= 0);
        assert("check valid launchParams ID" && launchParamsID <  launchParams.size());
        LaunchParams *launchParams = this->launchParams[launchParamsID];
        assert("check valid launchParams" && launchParams != nullptr);
        return launchParams;
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
      GeomGroup *checkGetGeomGroup(int groupID)
      {
        Group *group = checkGetGroup(groupID);
        assert("check valid group" && group != nullptr);
        GeomGroup *gg = dynamic_cast<GeomGroup*>(group);
        assert("check group is a geom group" && gg != nullptr);
        return gg;
      }
      // accessor helpers:
      UserGeomGroup *checkGetUserGeomGroup(int groupID)
      {
        Group *group = checkGetGroup(groupID);
        assert("check valid group" && group != nullptr);
        UserGeomGroup *ugg = dynamic_cast<UserGeomGroup*>(group);
        if (!ugg)
          OWL_EXCEPT("group is not a user geometry group");
        assert("check group is a user geom group" && ugg != nullptr);
        return ugg;
      }
      // accessor helpers:
      InstanceGroup *checkGetInstanceGroup(int groupID)
      {
        Group *group = checkGetGroup(groupID);
        InstanceGroup *ig = dynamic_cast<InstanceGroup*>(group);
        if (!ig)
          OWL_EXCEPT("group is not a instance group");
        assert("check group is a user geom group" && ig != nullptr);
        return ig;
      }
      
      Buffer *checkGetBuffer(int bufferID)
      {
        assert("check valid buffer ID" && bufferID >= 0);
        assert("check valid buffer ID" && bufferID <  buffers.size());
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
                                     LLOWriteUserGeomBoundsDataCB cb,
                                     const void *cbData);
      void sbtHitProgsBuild(LLOWriteHitProgDataCB writeHitProgDataCB,
                            const void *callBackUserData);
      void sbtRayGensBuild(LLOWriteRayGenDataCB writeRayGenDataCB,
                           const void *callBackUserData);
      void sbtMissProgsBuild(LLOWriteMissProgDataCB writeMissProgDataCB,
                             const void *callBackUserData);

      void launch(int rgID, const vec2i &dims);

      void launch(int rgID,
                  const vec2i &dims,
                  int32_t launchParamsID,
                  LLOWriteLaunchParamsCB writeLaunchParamsCB,
                  const void *cbData);

      void setRayTypeCount(size_t rayTypeCount);
      
      Context                  *context;
      
      Modules                     modules;
      std::vector<GeomType>       geomTypes;
      std::vector<RayGenPG>       rayGenPGs;
      std::vector<MissProgPG>     missProgPGs;
      std::vector<LaunchParams *> launchParams;
      std::vector<Geom *>         geoms;
      std::vector<Group *>        groups;
      std::vector<Buffer *>       buffers;
      // std::vector<cudaTextureObject_t> textureObjects;
      SBT                         sbt;
    };
    
  } // ::owl::ll
} //::owl
