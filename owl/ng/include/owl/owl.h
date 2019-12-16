
#pragma once

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
# include <cstddef> // is this c++??
#endif
#include "optix.h"


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


typedef enum
  {
   OWL_FLOAT=100,
   OWL_FLOAT2,
   OWL_FLOAT3,
   OWL_FLOAT4,

   OWL_INT=110,
   OWL_INT2,
   OWL_INT3,
   OWL_INT4,
   
   OWL_UINT=120,
   OWL_UINT2,
   OWL_UINT3,
   OWL_UINT4,
   
   OWL_BUFFER=1000,
   OWL_BUFFER_SIZE,
   OWL_BUFFER_ID,
   OWL_BUFFER_POINTER,
   OWL_BUFPTR=OWL_BUFFER_POINTER,

   OWL_GROUP=3000,
  }
  OWLDataType;

typedef enum
  {
   OWL_GEOMETRY_USER,
   OWL_GEOMETRY_TRIANGLES,
   OWL_TRIANGLES=OWL_GEOMETRY_TRIANGLES
  }
  OWLGeomKind;

#define OWL_ALL_RAY_TYPES -1


typedef float OWL_float;
typedef int32_t OWL_int;

typedef struct _OWL_int2   { int32_t x,y; } OWL_int2;
typedef struct _OWL_float2 { float   x,y; } OWL_float2;

typedef struct _OWL_int3   { int32_t x,y,z; } OWL_int3;
typedef struct _OWL_float3 { float   x,y,z; } OWL_float3;

typedef struct _OWL_int4   { int32_t x,y,z,w; } OWL_int4;
typedef struct _OWL_float4 { float   x,y,z,w; } OWL_float4;

typedef struct _OWLVarDecl {
  const char *name;
  OWLDataType type;
  uint32_t    offset;
} OWLVarDecl;

// ------------------------------------------------------------------
// device-objects - size of those _HAS_ to match the device-side
// definition of these types
// ------------------------------------------------------------------
typedef OptixTraversableHandle OWLDeviceTraversable;
typedef struct _OWLDeviceBuffer2D { void *d_pointer; OWL_int2 dims; } OWLDeviceBuffer2D;


typedef struct _OWLContext       *OWLContext;
typedef struct _OWLBuffer        *OWLBuffer;
typedef struct _OWLGeom          *OWLGeom;
typedef struct _OWLGeomType      *OWLGeomType;
typedef struct _OWLVariable      *OWLVariable;
typedef struct _OWLModule        *OWLModule;
typedef struct _OWLGroup         *OWLGroup;
typedef struct _OWLRayGen        *OWLRayGen;
typedef struct _OWLMissProg      *OWLMissProg;

// typedef OWLGeom OWLTriangles;

OWL_API void owlBuildPrograms(OWLContext context);
OWL_API void owlBuildPipeline(OWLContext context);


OWL_API OWLContext
owlContextCreate();

OWL_API void
owlContextDestroy(OWLContext context);

OWL_API OWLModule
owlModuleCreate(OWLContext context,
                const char *ptxCode);

OWL_API OWLGeom
owlContextCreateGeom(OWLContext context,
                     OWLGeomType type);

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

OWL_API OWLGroup
owlGeomGroupCreate(OWLContext context,
                   size_t     numGeometries,
                   OWLGeom   *initValues);

OWL_API OWLGroup
owlInstanceGroupCreate(OWLContext context,
                       size_t numInstances);

OWL_API OWLGeomType
owlGeomTypeCreate(OWLContext context,
                  OWLGeomKind kind,
                  size_t sizeOfVarStruct,
                  OWLVarDecl *vars,
                  size_t      numVars);

OWL_API OWLBuffer
owlDeviceBufferCreate(OWLContext context,
                      OWLDataType type,
                      int num,
                      const void *init);
OWL_API OWLBuffer
owlHostPinnedBufferCreate(OWLContext context,
                          OWLDataType type,
                          int num);

/*! executes an optix lauch of given size, with given launch
  program. Note this call is asynchronous, and may _not_ be
  completed by the time this function returns. */
OWL_API void
owlContextLaunch2D(OWLContext context,
                   OWLRayGen rayGen,
                   int dims_x, int dims_y);

// OWL_API OWLTriangles owlTrianglesCreate(OWLContext context,
//                                         size_t varsStructSize);

// ==================================================================
// "Triangles" functions
// ==================================================================
OWL_API void owlTrianglesSetVertices(OWLGeom triangles,
                                     OWLBuffer vertices);
OWL_API void owlTrianglesSetIndices(OWLGeom triangles,
                                    OWLBuffer indices);

// -------------------------------------------------------
// group/hierarchy creation and setting
// -------------------------------------------------------
OWL_API void
owlInstanceGroupSetChild(OWLGroup group,
                         int whichChild,
                         OWLGroup child);

OWL_API void
owlGeomTypeSetClosestHit(OWLGeomType type,
                         int rayType,
                         OWLModule module,
                         const char *progName);


// -------------------------------------------------------
// Release for the various types
// -------------------------------------------------------
OWL_API void owlGeomRelease(OWLGeom geometry);
OWL_API void owlVariableRelease(OWLVariable variable);
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

// -------------------------------------------------------
// VariableSet for different variable types
// -------------------------------------------------------
OWL_API void owlVariableSet1f(OWLVariable variable, const float value);
OWL_API void owlVariableSet3fv(OWLVariable variable, const float *value);
OWL_API void owlVariableSetGroup(OWLVariable variable, OWLGroup value);
OWL_API void owlVariableSetBuffer(OWLVariable variable, OWLBuffer value);

