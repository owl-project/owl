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

#include <cuda.h>
#include <driver_types.h>
#include <optix.h>

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
# include <cstddef> 
#endif


#if defined(_MSC_VER)
#  define OWL_DLL_EXPORT __declspec(dllexport)
#  define OWL_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_DLL_EXPORT
#  define OWL_DLL_IMPORT
#endif

#ifdef __cplusplus
# define OWL_IF_CPP(a) a
#else
# define OWL_IF_CPP(a) /* drop it */
#endif

#if defined(OWL_DLL_INTERFACE)
#  ifdef owl_EXPORTS
#    define OWL_API OWL_DLL_EXPORT
#  else
#    define OWL_API OWL_DLL_IMPORT
#  endif
#else
#  ifdef __cplusplus
#    define OWL_API extern "C" OWL_DLL_EXPORT
#  else
#    define OWL_API /* bla */
#  endif
//#  define OWL_API /*static lib*/
#endif
//#ifdef __cplusplus
//# define OWL_API extern "C" OWL_DLL_EXPORT
//#else
//# define OWL_API /* bla */
//#endif



#define OWL_OFFSETOF(type,member)               \
  ((char *)(&((struct type *)0)-> member )      \
   -                                            \
   (char *)(((struct type *)0)))
  
  
/*! enum that specifies the different possible memory layouts for
  passing transformation matrices */
typedef enum
  {
   /*! 4x3-float column-major matrix format, where a matrix is
     specified through four vec3fs, the first three being the basis
     vectors of the linear transform, and the fourth one the
     translation part. This is exactly the same layout as used in
     owl::common::affine3f (owl/common/math/AffineSpae.h) */
   OWL_MATRIX_FORMAT_COLUMN_MAJOR=0,
   
   /*! just another name for OWL_MATRIX_FORMAT_4X3_COLUMN_MAJOR that
     is easier to type - the "_OWL" indicates that this is the default
     format in the owl::common namespace */
   OWL_MATRIX_FORMAT_OWL=OWL_MATRIX_FORMAT_COLUMN_MAJOR,
   
   /*! 3x4-float *row-major* layout as preferred by optix; in this
     case it doesn't matter if it's a 4x3 or 4x4 matrix, since the
     last row in a 4x4 row major matrix can simply be ignored */
   OWL_MATRIX_FORMAT_ROW_MAJOR
  } OWLMatrixFormat;

typedef enum
  {
   OWL_SBT_HITGROUPS = 0x1,
   OWL_SBT_GEOMS     = OWL_SBT_HITGROUPS,
   OWL_SBT_RAYGENS   = 0x2,
   OWL_SBT_MISSPROGS = 0x4,
   OWL_SBT_ALL   = 0x7
  } OWLBuildSBTFlags;
  
typedef enum
  {
   OWL_INVALID_TYPE = 0,

   OWL_BUFFER=10,
   OWL_BUFFER_SIZE,
   OWL_BUFFER_ID,
   OWL_BUFFER_POINTER,
   OWL_BUFPTR=OWL_BUFFER_POINTER,

   OWL_GROUP=20,

   /*! implicit variable of type integer that specifies the *index*
     of the given device. this variable type is implicit in the
     sense that it only gets _declared_ on the host, and gets set
     automatically during SBT creation */
   OWL_DEVICE=30,

   /*! texture(s) */
   OWL_TEXTURE=40,
   OWL_TEXTURE_2D=OWL_TEXTURE,


   /* all types that are naively copyable should be below this value,
      all that aren't should be above */
   _OWL_BEGIN_COPYABLE_TYPES = 1000,
   
   
   OWL_FLOAT=1000,
   OWL_FLOAT2,
   OWL_FLOAT3,
   OWL_FLOAT4,

   OWL_INT=1100,
   OWL_INT2,
   OWL_INT3,
   OWL_INT4,
   
   OWL_UINT=1200,
   OWL_UINT2,
   OWL_UINT3,
   OWL_UINT4,
   
   OWL_LONG=1300,
   OWL_LONG2,
   OWL_LONG3,
   OWL_LONG4,

   OWL_ULONG=1400,
   OWL_ULONG2,
   OWL_ULONG3,
   OWL_ULONG4,

   OWL_DOUBLE=1500,
   OWL_DOUBLE2,
   OWL_DOUBLE3,
   OWL_DOUBLE4,
    
   OWL_CHAR=1600,
   OWL_CHAR2,
   OWL_CHAR3,
   OWL_CHAR4,

   /*! unsigend 8-bit integer */
   OWL_UCHAR=1700,
   OWL_UCHAR2,
   OWL_UCHAR3,
   OWL_UCHAR4,

   /*! just another name for a 64-bit data type - unlike
     OWL_BUFFER_POINTER's (which gets translated from OWLBuffer's
     to actual device-side poiners) these OWL_RAW_POINTER types get
     copied binary without any translation. This is useful for
     owl-cuda interaction (where the user already has device
     pointers), but should not be used for logical buffers */
   OWL_RAW_POINTER=OWL_ULONG,


   /* matrix formats */
   OWL_AFFINE3F=1800,

   /*! at least for now, use that for buffers with user-defined types:
     type then is "OWL_USER_TYPE_BEGIN+sizeof(elementtype). Note
     that since we always _add_ the user type's size to this value
     this MUST be the last entry in the enum */
   OWL_USER_TYPE_BEGIN=10000
  }
  OWLDataType;

#define OWL_USER_TYPE(userType) ((OWLDataType)(OWL_USER_TYPE_BEGIN+sizeof(userType)))

typedef enum
  {
   // soon to be deprecated old naming
   OWL_GEOMETRY_USER,
   // new naming, to be consistent with type OLWGeom (not OWLGeometry):
   OWL_GEOM_USER=OWL_GEOMETRY_USER,
   // soon to be deprecated old naming
   OWL_GEOMETRY_TRIANGLES,
   // new naming, to be consistent with type OLWGeom (not OWLGeometry):
   OWL_GEOM_TRIANGLES=OWL_GEOMETRY_TRIANGLES,
   OWL_TRIANGLES=OWL_GEOMETRY_TRIANGLES,
   OWL_GEOMETRY_HAIR
  }
  OWLGeomKind;

#define OWL_ALL_RAY_TYPES -1


typedef float    OWL_float;
typedef double   OWL_double;
typedef int32_t  OWL_int;
typedef uint32_t OWL_uint;
typedef int64_t  OWL_long;
typedef uint64_t OWL_ulong;

typedef struct _OWL_int2    { int32_t  x,y; } owl2i;
typedef struct _OWL_uint2   { int32_t  x,y; } owl2ui;
typedef struct _OWL_long2   { int64_t  x,y; } owl2l;
typedef struct _OWL_ulong2  { uint64_t x,y; } owl2ul;
typedef struct _OWL_float2  { float    x,y; } owl2f;
typedef struct _OWL_double2 { double   x,y; } owl2d;

typedef struct _OWL_int3    { int32_t  x,y,z; } owl3i;
typedef struct _OWL_uint3   { uint32_t x,y,z; } owl3ui;
typedef struct _OWL_long3   { int64_t  x,y,z; } owl3l;
typedef struct _OWL_ulong3  { uint64_t x,y,z; } owl3ul;
typedef struct _OWL_float3  { float    x,y,z; } owl3f;
typedef struct _OWL_double3 { double   x,y,z; } owl3d;

typedef struct _OWL_int4    { int32_t  x,y,z,w; } owl4i;
typedef struct _OWL_uint4   { uint32_t x,y,z,w; } owl4ui;
typedef struct _OWL_long4   { int64_t  x,y,z,w; } owl4l;
typedef struct _OWL_ulong4  { uint64_t x,y,z,w; } owl4ul;
typedef struct _OWL_float4  { float    x,y,z,w; } owl4f;
typedef struct _OWL_double4 { double   x,y,z,w; } owl4d;

typedef struct _OWL_affine3f { owl3f vx,vy,vz,t; } owl4x3f;

typedef struct _OWLVarDecl {
  const char *name;
  OWLDataType type;
  uint32_t    offset;
} OWLVarDecl;


/*! supported formats for texels in textures */
typedef enum
  {
   OWL_TEXEL_FORMAT_RGBA8
  }
  OWLTexelFormat;

/*! currently supported texture filter modes */
typedef enum
  {
   OWL_TEXTURE_NEAREST,
   OWL_TEXTURE_LINEAR
  }
  OWLTextureFilterMode;
  


// ------------------------------------------------------------------
// device-objects - size of those _HAS_ to match the device-side
// definition of these types
// ------------------------------------------------------------------
typedef OptixTraversableHandle OWLDeviceTraversable;
typedef struct _OWLDeviceBuffer2D { void *d_pointer; owl2i dims; } OWLDeviceBuffer2D;


typedef struct _OWLContext       *OWLContext;
typedef struct _OWLBuffer        *OWLBuffer;
typedef struct _OWLTexture       *OWLTexture;
typedef struct _OWLGeom          *OWLGeom;
typedef struct _OWLGeomType      *OWLGeomType;
typedef struct _OWLVariable      *OWLVariable;
typedef struct _OWLModule        *OWLModule;
typedef struct _OWLGroup         *OWLGroup;
typedef struct _OWLRayGen        *OWLRayGen;
typedef struct _OWLMissProg      *OWLMissProg;
/*! launch params (or "globals") are variables that can be put into
  device constant memory, accessible through a CUDA "__constant__
  <Type> optixLaunchParams;" variable on the device side. Launch
  params capture the layout of this struct, and the value of its
  members, on the host side, then properly fill it in before executing
  a launch. OptiX calls those "launch parameters" because they are
  similar to how parameters to a CUDA kernel are internally treated;
  we also call them "globals" because they are globally accessible to
  all programs within a given launch */
typedef struct _OWLLaunchParams  *OWLLaunchParams, *OWLParams, *OWLGlobals;

OWL_API void owlBuildPrograms(OWLContext context);
OWL_API void owlBuildPipeline(OWLContext context);
OWL_API void owlBuildSBT(OWLContext context,
                         OWLBuildSBTFlags flags OWL_IF_CPP(=OWL_SBT_ALL));

/*! returns number of devices available in the given context */
OWL_API int32_t
owlGetDeviceCount(OWLContext context);

/*! creates a new device context with the gives list of devices. 

  If requested device IDs list if null it implicitly refers to the
  list "0,1,2,...."; if numDevices <= 0 it automatically refers to
  "all devices you can find". Examples:

  - owlContextCreate(nullptr,1) creates one device on the first GPU

  - owlContextCreate(nullptr,0) creates a context across all GPUs in
  the system

  - int gpu=2;owlContextCreate(&gpu,1) will create a context on GPU #2
  (where 2 refers to the CUDA device ordinal; from that point on, from
  owl's standpoint (eg, during owlBufferGetPointer() this GPU will
  from that point on be known as device #0 */
OWL_API OWLContext
owlContextCreate(int32_t *requestedDeviceIDs OWL_IF_CPP(=nullptr),
                 int numDevices OWL_IF_CPP(=0));

/*! enable motion blur for this context. this _has_ to be called
    before creating any geometries, groups, etc, and before the
    pipeline gets compiled. Ie, it shold be called _right_ after
    context creation */
OWL_API void
owlEnableMotionBlur(OWLContext _context);

/*! set number of ray types to be used in this context; this should be
  done before any programs, pipelines, geometries, etc get
  created */
OWL_API void
owlContextSetRayTypeCount(OWLContext context,
                          size_t numRayTypes);

/*! sets maximum instancing depth for the given context:

  '0' means 'no instancing allowed, only bottom-level accels; Note
  this mode isn't actually allowed in OWL right now, as the most
  convenient way of realizing it is actually *slower* than simply
  putting a single "dummy" instance (with just this one child, and a
  identify transform) over each blas.

  '1' means 'at most one layer of instances' (ie, a two-level scene),
  where the 'root' world rays are traced against can be an instance
  group, but every child in that inscne group is a geometry group.

  'N>1" means "up to N layers of instances are allowed.

  The default instancing depth is 1 (ie, a two-level scene), since
  this allows for most use cases of instancing and is still
  hardware-accelerated. Using a node graph with instancing deeper than
  the configured value will result in wrong results; but be aware that
  using any value > 1 here will come with a cost. It is recommended
  to, if at all possible, leave this value to one and convert the
  input scene to a two-level scene layout (ie, with only one level of
  instances) */
OWL_API void
owlSetMaxInstancingDepth(OWLContext context,
                         int32_t maxInstanceDepth);
  

OWL_API void
owlContextDestroy(OWLContext context);

/* return the cuda stream associated with the given device. */
OWL_API CUstream
owlContextGetStream(OWLContext context, int deviceID);

/* return the optix context associated with the given device. */
OWL_API OptixDeviceContext
owlContextGetOptixContext(OWLContext context, int deviceID);

OWL_API OWLModule
owlModuleCreate(OWLContext  context,
                const char *ptxCode);

OWL_API OWLGeom
owlGeomCreate(OWLContext  context,
              OWLGeomType type);

OWL_API OWLParams
owlParamsCreate(OWLContext  context,
                size_t      sizeOfVarStruct,
                OWLVarDecl *vars,
                size_t      numVars);

OWL_API OWLRayGen
owlRayGenCreate(OWLContext  context,
                OWLModule   module,
                const char *programName,
                size_t      sizeOfVarStruct,
                OWLVarDecl *vars,
                size_t      numVars);

OWL_API OWLMissProg
owlMissProgCreate(OWLContext  context,
                  OWLModule   module,
                  const char *programName,
                  size_t      sizeOfVarStruct,
                  OWLVarDecl *vars,
                  size_t      numVars);


// ------------------------------------------------------------------
/*! create a new group (which handles the acceleration strucure) for
  triangle geometries.

  \param numGeometries Number of geometries in this group, must be
  non-zero.

  \param arrayOfChildGeoms A array of 'numGeomteries' child
  geometries. Every geom in this array must be a valid owl geometry
  created with owlGeomCreate, and must be of a OWL_GEOM_USER
  type.
*/
OWL_API OWLGroup
owlUserGeomGroupCreate(OWLContext context,
                       size_t     numGeometries,
                       OWLGeom   *arrayOfChildGeoms);


// ------------------------------------------------------------------
/*! create a new group (which handles the acceleration strucure) for
  triangle geometries.

  \param numGeometries Number of geometries in this group, must be
  non-zero.

  \param arrayOfChildGeoms A array of 'numGeometries' child
  geometries. Every geom in this array must be a valid owl geometry
  created with owlGeomCreate, and must be of a OWL_GEOM_TRIANGLES
  type.
*/
OWL_API OWLGroup
owlTrianglesGeomGroupCreate(OWLContext context,
                            size_t     numGeometries,
                            OWLGeom   *initValues);

// ------------------------------------------------------------------
/*! create a new instance group with given number of instances. The
  child groups and their instance IDs and/or transforms can either
  be specified "in bulk" as part of this call, or can be set lateron
  with inviidaul calls to \see owlInstanceGroupSetChild and \see
  owlInstanceGroupSetTransform. Note however, that in the case of
  having millions of instances in a group it will be *much* more
  efficient to set them in bulk open creation, than in millions of
  inidiviual API calls.

  Either or all of initGroups, initTranforms, or initInstanceIDs may
  be null, in which case the values used for the 'th child will be a
  null group, a unit transform, and 'i', respectively.
*/
OWL_API OWLGroup
owlInstanceGroupCreate(OWLContext context,
                       
                       /*! number of instances in this group */
                       size_t     numInstances,
                       
                       /*! the initial list of owl groups to use by
                         the instances in this group; must be either
                         null, or an array of the size
                         'numInstnaces', the i'th instnace in this
                         gorup will be an instance o the i'th
                         element in this list */
                       const OWLGroup *initGroups      OWL_IF_CPP(= nullptr),

                       /*! instance IDs to use for the instance in
                         this group; must be eithe rnull, or an
                         array of size numInstnaces. If null, the
                         i'th child of this instance group will use
                         instanceID=i, otherwise, it will use the
                         user-provided instnace ID from this
                         list. Specifying an instanceID will affect
                         what value 'optixGetInstanceID' will return
                         in a CH program that refers to the given
                         instance */
                       const uint32_t *initInstanceIDs OWL_IF_CPP(= nullptr),
                       
                       /*! initial list of transforms that this
                         instance group will use; must be either
                         null, or an array of size numInstnaces, of
                         the format specified */
                       const float    *initTransforms  OWL_IF_CPP(= nullptr),
                       OWLMatrixFormat matrixFormat    OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL)
                       );



OWL_API void owlGroupBuildAccel(OWLGroup group);
OWL_API void owlGroupRefitAccel(OWLGroup group);

OWL_API OWLGeomType
owlGeomTypeCreate(OWLContext context,
                  OWLGeomKind kind,
                  size_t sizeOfVarStruct,
                  OWLVarDecl *vars,
                  size_t      numVars);


/*! create new texture of given format and dimensions - for now, we
  only do "wrap" textures, and eithe rbilinear or nearest filter;
  once we allow for doing things like texture borders we'll have to
  change this api */
OWL_API OWLTexture
owlTexture2DCreate(OWLContext context,
                   OWLTexelFormat texelFormat,
                   /*! number of texels in x dimension */
                   uint32_t size_x,
                   /*! number of texels in y dimension */
                   uint32_t size_y,
                   const void *texels,
                   OWLTextureFilterMode filterMode OWL_IF_CPP(=OWL_TEXTURE_LINEAR),
                   /*! number of bytes between one line of texels and
                     the next; '0' means 'size_x * sizeof(texel)' */
                   uint32_t linePitchInBytes       OWL_IF_CPP(=0)
                   );

/*! destroy the given texture; after this call any accesses to the given texture are invalid */
OWL_API void
owlTexture2DDestroy(OWLTexture texture);

/*! creates a device buffer where every device has its own local copy
  of the given buffer */
OWL_API OWLBuffer
owlDeviceBufferCreate(OWLContext  context,
                      OWLDataType type,
                      size_t      count,
                      const void *init);

/*! creates a buffer that uses CUDA host pinned memory; that memory is
  pinned on the host and accessive to all devices in the deviec
  group */
OWL_API OWLBuffer
owlHostPinnedBufferCreate(OWLContext context,
                          OWLDataType type,
                          size_t      count);

/*! creates a buffer that uses CUDA managed memory; that memory is
  managed by CUDA (see CUDAs documentatoin on managed memory) and
  accessive to all devices in the deviec group */
OWL_API OWLBuffer
owlManagedMemoryBufferCreate(OWLContext context,
                             OWLDataType type,
                             size_t      count,
                             const void *init);

/*! creates a buffer wrapping a CUDA graphics resource;
  the resource must be created and registered by the user */
OWL_API OWLBuffer
owlGraphicsBufferCreate(OWLContext             context,
                        OWLDataType            type,
                        size_t                 count,
                        cudaGraphicsResource_t resource);

OWL_API void
owlGraphicsBufferMap(OWLBuffer buffer);

OWL_API void
owlGraphicsBufferUnmap(OWLBuffer buffer);

/*! returns the device pointer of the given pointer for the given
  device ID. For host-pinned or managed memory buffers (where the
  buffer is shared across all devices) this pointer should be the
  same across all devices (and even be accessible on the host); for
  device buffers each device *may* see this buffer under a different
  address, and that address is not valid on the host. Note this
  function is paricuarly useful for CUDA-interop; allowing to
  cudaMemcpy to/from an owl buffer directly from CUDA code */
OWL_API const void *
owlBufferGetPointer(OWLBuffer buffer, int deviceID);

OWL_API OptixTraversableHandle 
owlGroupGetTraversable(OWLGroup group, int deviceID);

OWL_API void 
owlBufferResize(OWLBuffer buffer, size_t newItemCount);

/*! destroy the given buffer; this will both release the app's
  refcount on the given buffer handle, *and* the buffer itself; ie,
  even if some objects still hold variables that refer to the old
  handle the buffer itself will be freed */
OWL_API void 
owlBufferDestroy(OWLBuffer buffer);

OWL_API void 
owlBufferUpload(OWLBuffer buffer, const void *hostPtr);

/*! executes an optix lauch of given size, with given launch
  program. Note this is asynchronous, and may _not_ be
  completed by the time this function returns. */
OWL_API void
owlRayGenLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y);

/*! perform a raygen launch with lauch parameters, in a *synchronous*
    way; it, by the time this function returns the launch is completed */
OWL_API void
owlLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y,
            OWLParams params);

/*! perform a raygen launch with lauch parameters, in a *A*synchronous
    way; it, this will only launch, but *NOT* wait for completion (see
    owlLaunchSync) */
OWL_API void
owlAsyncLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y,
                 OWLParams params);


OWL_API CUstream
owlParamsGetCudaStream(OWLParams params, int deviceID);

/*! wait for the async launch to finish */
OWL_API void
owlLaunchSync(OWLParams params);

// ==================================================================
// "Triangles" functions
// ==================================================================
OWL_API void owlTrianglesSetVertices(OWLGeom triangles,
                                     OWLBuffer vertices,
                                     size_t count,
                                     size_t stride,
                                     size_t offset);
OWL_API void owlTrianglesSetMotionVertices(OWLGeom triangles,
                                           /*! number of vertex arrays
                                               passed here, the first
                                               of those is for t=0,
                                               thelast for t=1,
                                               everything is linearly
                                               interpolated
                                               in-between */
                                           size_t    numKeys,
                                           OWLBuffer *vertexArrays,
                                           size_t count,
                                           size_t stride,
                                           size_t offset);
OWL_API void owlTrianglesSetIndices(OWLGeom triangles,
                                    OWLBuffer indices,
                                    size_t count,
                                    size_t stride,
                                    size_t offset);

// -------------------------------------------------------
// group/hierarchy creation and setting
// -------------------------------------------------------
OWL_API void
owlInstanceGroupSetChild(OWLGroup group,
                         int whichChild,
                         OWLGroup child);

/*! sets the transformatoin matrix to be applied to the childID'th
  child of the given instance group */
OWL_API void
owlInstanceGroupSetTransform(OWLGroup group,
                             int whichChild,
                             const float *floats,
                             OWLMatrixFormat matrixFormat    OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL));

/*! this function allows to set up to N different arrays of trnsforms
    for motion blur; the first such array is used as transforms for
    t=0, the last one for t=1.  */
OWL_API void
owlInstanceGroupSetTransforms(OWLGroup group,
                              /*! whether to set for t=0 or t=1 -
                                  currently supporting only 0 or 1*/
                              uint32_t timeStep,
                              const float *floatsForThisStimeStep,
                              OWLMatrixFormat matrixFormat    OWL_IF_CPP(=OWL_MATRIX_FORMAT_OWL));

/*! sets the list of IDs to use for the child instnaces. By default
    the instance ID of child #i is simply i, but optix allows to
    specify a user-defined instnace ID for each instance, which with
    owl can be done through this array. Array size must match number
    of instances in the specified group */
OWL_API void
owlInstanceGroupSetInstanceIDs(OWLGroup group,
                               const uint32_t *instanceIDs);

OWL_API void
owlGeomTypeSetClosestHit(OWLGeomType type,
                         int rayType,
                         OWLModule module,
                         const char *progName);

OWL_API void
owlGeomTypeSetAnyHit(OWLGeomType type,
                     int rayType,
                     OWLModule module,
                     const char *progName);

OWL_API void
owlGeomTypeSetIntersectProg(OWLGeomType type,
                            int rayType,
                            OWLModule module,
                            const char *progName);

OWL_API void
owlGeomTypeSetBoundsProg(OWLGeomType type,
                         OWLModule module,
                         const char *progName);

/*! set the primitive count for the given uesr geometry. this _has_ to
  be set before the group(s) that this geom is used in get built */
OWL_API void
owlGeomSetPrimCount(OWLGeom geom,
                    size_t  primCount);


// -------------------------------------------------------
// Release for the various types
// -------------------------------------------------------
OWL_API void owlGeomRelease(OWLGeom geometry);
OWL_API void owlVariableRelease(OWLVariable variable);
OWL_API void owlModuleRelease(OWLModule module);
OWL_API void owlBufferRelease(OWLBuffer buffer);
OWL_API void owlRayGenRelease(OWLRayGen rayGen);
OWL_API void owlGroupRelease(OWLGroup group);

// -------------------------------------------------------
// VariableGet for the various types
// -------------------------------------------------------
OWL_API OWLVariable
owlGeomGetVariable(OWLGeom geom,
                   const char *varName);

OWL_API OWLVariable
owlRayGenGetVariable(OWLRayGen geom,
                     const char *varName);

OWL_API OWLVariable
owlMissProgGetVariable(OWLMissProg geom,
                       const char *varName);

OWL_API OWLVariable
owlParamsGetVariable(OWLParams object,
                     const char *varName);

// -------------------------------------------------------
// VariableSet for different variable types
// -------------------------------------------------------

OWL_API void owlVariableSetGroup(OWLVariable variable, OWLGroup value);
OWL_API void owlVariableSetTexture(OWLVariable variable, OWLTexture value);
OWL_API void owlVariableSetBuffer(OWLVariable variable, OWLBuffer value);
OWL_API void owlVariableSetRaw(OWLVariable variable, const void *valuePtr);
OWL_API void owlVariableSetPointer(OWLVariable variable, const void *valuePtr);
#define _OWL_SET_HELPER(stype,abb)                      \
  OWL_API void owlVariableSet1##abb(OWLVariable var,    \
                                    stype v);           \
  OWL_API void owlVariableSet2##abb(OWLVariable var,    \
                                    stype x,            \
                                    stype y);           \
  OWL_API void owlVariableSet3##abb(OWLVariable var,    \
                                    stype x,            \
                                    stype y,            \
                                    stype z);           \
  /*end of macro */
_OWL_SET_HELPER(int32_t,i)
_OWL_SET_HELPER(uint32_t,ui)
_OWL_SET_HELPER(int64_t,l)
_OWL_SET_HELPER(uint64_t,ul)
_OWL_SET_HELPER(float,f)
_OWL_SET_HELPER(double,d)
#undef _OWL_SET_HELPER





// -------------------------------------------------------
// VariableSet for different *object* types
// -------------------------------------------------------

#define _OWL_SET_HELPERS_C(OType,stype,abb)                     \
  /* set1 */                                                    \
  inline void owl##OType##Set1##abb(OWL##OType object,          \
                                    const char *varName,        \
                                    stype v)                    \
  {                                                             \
    OWLVariable var                                             \
      = owl##OType##GetVariable(object,varName);                \
    owlVariableSet1##abb(var,v);                                \
    owlVariableRelease(var);                                    \
  }                                                             \
  /* set2 */                                                    \
  inline void owl##OType##Set2##abb(OWL##OType object,          \
                                    const char *varName,        \
                                    stype x,                    \
                                    stype y)                    \
  {                                                             \
    OWLVariable var                                             \
      = owl##OType##GetVariable(object,varName);                \
    owlVariableSet2##abb(var,x,y);                              \
    owlVariableRelease(var);                                    \
  }                                                             \
  /* set3 */                                                    \
  inline void owl##OType##Set3##abb(OWL##OType object,          \
                                    const char *varName,        \
                                    stype x,                    \
                                    stype y,                    \
                                    stype z)                    \
  {                                                             \
    OWLVariable var                                             \
      = owl##OType##GetVariable(object,varName);                \
    owlVariableSet3##abb(var,x,y,z);                            \
    owlVariableRelease(var);                                    \
  }                                                             \
  /* end of macro */


#ifdef __cplusplus
#define _OWL_SET_HELPERS_CPP(OType,stype,abb)                   \
  inline void owl##OType##Set2##abb(OWL##OType object,          \
                                    const char *varName,        \
                                    const owl2##abb &v)         \
  {                                                             \
    OWLVariable var                                             \
      = owl##OType##GetVariable(object,varName);                \
    owlVariableSet2##abb(var,v.x,v.y);                          \
    owlVariableRelease(var);                                    \
  }                                                             \
  inline void owl##OType##Set3##abb(OWL##OType object,          \
                                    const char *varName,        \
                                    const owl3##abb &v)         \
  {                                                             \
    OWLVariable var                                             \
      = owl##OType##GetVariable(object,varName);                \
    owlVariableSet3##abb(var,v.x,v.y,v.z);                      \
    owlVariableRelease(var);                                    \
  }                                                             \
  /* end of macro */
#else
#define _OWL_SET_HELPERS_CPP(OType,stype,abb)  /* ignore in C99 mode */
#endif

#define _OWL_SET_HELPERS(Type)                                  \
  /* texture, buffer, other */                                  \
  inline void owl##Type##SetTexture(OWL##Type object,           \
                                    const char *varName,        \
                                    OWLTexture v)               \
  {                                                             \
    OWLVariable var                                             \
      = owl##Type##GetVariable(object,varName);                 \
    owlVariableSetTexture(var,v);                               \
    owlVariableRelease(var);                                    \
  }                                                             \
  /* group, buffer, other */                                    \
  inline void owl##Type##SetGroup(OWL##Type object,             \
                                  const char *varName,          \
                                  OWLGroup v)                   \
  {                                                             \
    OWLVariable var                                             \
      = owl##Type##GetVariable(object,varName);                 \
    owlVariableSetGroup(var,v);                                 \
    owlVariableRelease(var);                                    \
  }                                                             \
  inline void owl##Type##SetRaw(OWL##Type object,               \
                                const char *varName,            \
                                const void *v)                  \
  {                                                             \
    OWLVariable var                                             \
      = owl##Type##GetVariable(object,varName);                 \
    owlVariableSetRaw(var,v);                                   \
    owlVariableRelease(var);                                    \
  }                                                             \
  inline void owl##Type##SetPointer(OWL##Type object,           \
                                    const char *varName,        \
                                    const void *v)              \
  {                                                             \
    OWLVariable var                                             \
      = owl##Type##GetVariable(object,varName);                 \
    owlVariableSetPointer(var,v);                               \
    owlVariableRelease(var);                                    \
  }                                                             \
  inline void owl##Type##SetBuffer(OWL##Type object,            \
                                   const char *varName,         \
                                   OWLBuffer v)                 \
  {                                                             \
    OWLVariable var                                             \
      = owl##Type##GetVariable(object,varName);                 \
    owlVariableSetBuffer(var,v);                                \
    owlVariableRelease(var);                                    \
  }                                                             \
                                                                \
  _OWL_SET_HELPERS_C(Type,int32_t,i)                            \
  _OWL_SET_HELPERS_C(Type,uint32_t,ui)                          \
  _OWL_SET_HELPERS_C(Type,int64_t,l)                            \
  _OWL_SET_HELPERS_C(Type,uint64_t,ul)                          \
  _OWL_SET_HELPERS_C(Type,float,f)                              \
  _OWL_SET_HELPERS_C(Type,double,d)                             \
  _OWL_SET_HELPERS_CPP(Type,int32_t,i)                          \
  _OWL_SET_HELPERS_CPP(Type,uint32_t,ui)                        \
  _OWL_SET_HELPERS_CPP(Type,int64_t,l)                          \
  _OWL_SET_HELPERS_CPP(Type,uint64_t,ul)                        \
  _OWL_SET_HELPERS_CPP(Type,float,f)                            \
  _OWL_SET_HELPERS_CPP(Type,double,d)                           \
  /* end of macro */

  _OWL_SET_HELPERS(RayGen)
  _OWL_SET_HELPERS(Geom)
  _OWL_SET_HELPERS(Params)
  _OWL_SET_HELPERS(MissProg)

#undef _OWL_SET_HELPERS_CPP
#undef _OWL_SET_HELPERS_C
#undef _OWL_SET_HELPERS


#ifdef __cplusplus
/*! c++ "convenience variant" of owlInstanceGroupSetTransform that
  also allows passing C++ types) */
  inline void
  owlInstanceGroupSetTransform(OWLGroup group,
                               int childID,
                               const owl4x3f &xfm)
  {
    owlInstanceGroupSetTransform(group,childID,(const float *)&xfm,
                                 OWL_MATRIX_FORMAT_OWL);
  }
/*! c++ "convenience variant" of owlInstanceGroupSetTransform that
  also allows passing C++ types) */
inline void
owlInstanceGroupSetTransform(OWLGroup group,
                             int childID,
                             const owl4x3f *xfm)
{
  owlInstanceGroupSetTransform(group,childID,(const float *)xfm,
                               OWL_MATRIX_FORMAT_OWL);
}

#endif










