
#pragma once

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
# include <cstddef> // is this c++??
#endif
#include "optix.h"

#define OWL_OFFSETOF(type,member)               \
  ((char *)(&((struct type *)0)-> member )             \
   -                                            \
   (char *)(((struct type *)0)))

#ifdef __cplusplus
# define OWL_API extern "C"
#else
# define OWL_API /* bla */
#endif

typedef enum
  {
   OWL_FLOAT3,
   OWL_INT3,
   OWL_BUFFER,
   OWL_BUFFER_SIZE,
   OWL_BUFFER_ID,
   OWL_BUFFER_POINTER
  }
  OWLDataType;

typedef enum
  {
   OWL_GEOMETRY_USER,
   OWL_GEOMETRY_TRIANGLES
  }
  OWLGeometryKind;

#define OWL_ALL_RAY_TYPES -1


typedef float OWL_float;
typedef int32_t OWL_int;

typedef struct _OWL_int2   { int32_t x,y; } OWL_int2;
typedef struct _OWL_float2 { float   x,y; } OWL_float2;

typedef struct _OWL_int3   { int32_t x,y,z; } OWL_int3;
typedef struct _OWL_float3 { float   x,y,z; } OWL_float3;


// ------------------------------------------------------------------
// device-objects - size of those _HAS_ to match the device-side
// definition of these types
// ------------------------------------------------------------------
typedef OptixTraversableHandle OWLDeviceTraversable;
typedef struct _OWLDeviceBuffer2D { void *d_pointer; OWL_int2 dims; } OWLDeviceBuffer2D;


typedef struct _OWLContext       *OWLContext;
typedef struct _OWLBuffer        *OWLBuffer;
typedef struct _OWLGeometry      *OWLGeometry;
typedef struct _OWLGeometryType  *OWLGeometryType;
typedef struct _OWLVariable      *OWLVariable;
typedef struct _OWLModule        *OWLModule;
typedef struct _OWLGeometryGroup *OWLGeometryGroup;
typedef struct _OWLInstanceGroup *OWLInstanceGroup;
typedef struct _OWLLaunchProg    *OWLLaunchProg;

// typedef OWLGeometry OWLTriangles;

OWL_API OWLContext
owlContextCreate();

OWL_API void
owlContextDestroy(OWLContext context);

OWL_API OWLModule
owlContextCreateModule(OWLContext context,
                       const char *ptxCode);

OWL_API OWLGeometry
owlContextCreateGeometry(OWLContext context,
                         OWLGeometryType type);

OWL_API OWLLaunchProg
owlContextCreateLaunchProg(OWLContext context,
                           OWLModule module,
                           const char *programName,
                           size_t sizeOfVarStruct);

OWL_API OWLGeometryGroup
owlContextCreateGeometryGroup(OWLContext context,
                              size_t numGeometries,
                              OWLGeometry *initValues);

OWL_API OWLInstanceGroup
owlContextCreateInstanceGroup(OWLContext context,
                              size_t numInstances);

OWL_API OWLGeometryType
owlContextCreateGeometryType(OWLContext context,
                             OWLGeometryKind kind,
                             size_t sizeOfVarStruct);

OWL_API OWLBuffer
owlContextCreateBuffer(OWLContext context,
                       OWLDataType type,
                       int num,
                       const void *init);

/*! executes an optix lauch of given size, with given launch
    program. Note this call is asynchronous, and may _not_ be
    completed by the time this function returns. */
OWL_API void
owlContextLaunch2D(OWLContext context,
                   OWLLaunchProg launchProg,
                   int dims_x, int dims_y);

// OWL_API OWLTriangles owlTrianglesCreate(OWLContext context,
//                                         size_t varsStructSize);

// ==================================================================
// "Triangles" functions
// ==================================================================
OWL_API void owlTrianglesSetVertices(OWLGeometry triangles,
                                     OWLBuffer vertices);
OWL_API void owlTrianglesSetIndices(OWLGeometry triangles,
                                    OWLBuffer indices);

OWL_API void
owlGeometryTypeDeclareVariable(OWLGeometryType object,
                               const char *varName,
                               OWLDataType type,
                               size_t offset);


// -------------------------------------------------------
// group/hierarchy creation and setting
// -------------------------------------------------------
OWL_API void
owlInstanceGroupSetChild(OWLInstanceGroup group,
                         int whichChild,
                         OWLGeometryGroup geometry);

OWL_API void
owlGeometryTypeSetClosestHitProgram(OWLGeometryType type,
                                    int rayType,
                                    OWLModule module,
                                    const char *progName);


// -------------------------------------------------------
// Release for the various types
// -------------------------------------------------------
OWL_API void owlGeometryRelease(OWLGeometry geometry);
OWL_API void owlVariableRelease(OWLVariable variable);
OWL_API void owlBufferRelease(OWLBuffer buffer);

// -------------------------------------------------------
// VariableGet for the various types
// -------------------------------------------------------
OWL_API OWLVariable
owlGeometryGetVariable(OWLGeometry geom,
                       const char *varName);

OWL_API OWLVariable
owlLaunchProgGetVariable(OWLLaunchProg geom,
                         const char *varName);

// -------------------------------------------------------
// VariableSet for different variable types
// -------------------------------------------------------
OWL_API void owlVariableSet1f(OWLVariable variable, const float value);
OWL_API void owlVariableSet3fv(OWLVariable variable, const float *value);

