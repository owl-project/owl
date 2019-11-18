
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
   OWL_TRIANGLES
  }
  OWLGeometryKind;



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


#ifdef __cplusplus
typedef struct _OWLObject {}   *OWLObject;
typedef struct _OWLContext   : public _OWLObject {} *OWLContext;
typedef struct _OWLBuffer    : public _OWLObject {} *OWLBuffer;
typedef struct _OWLVariable  : public _OWLObject {} *OWLVariable;
typedef struct _OWLModule    : public _OWLObject {} *OWLModule;
typedef struct _OWLGeometryGroup : public _OWLObject {} *OWLGeometryGroup;
typedef struct _OWLInstanceGroup : public _OWLObject {} *OWLInstanceGroup;
typedef struct _OWLGeometry    : public _OWLObject {} *OWLGeometry;
typedef struct _OWLGeometryType    : public _OWLObject {} *OWLGeometryType;
#else
typedef struct _OWLObject    *OWLObject;
typedef struct _OWLContext   *OWLContext;
typedef struct _OWLBuffer    *OWLBuffer;
typedef struct _OWLGeometry    *OWLGeometry;
typedef struct _OWLGeometryType    *OWLGeometryType;
typedef struct _OWLVariable  *OWLVariable;
typedef struct _OWLModule    *OWLModule;
typedef struct _OWLGeometryGroup    *OWLGeometryGroup;
typedef struct _OWLInstanceGroup    *OWLInstanceGroup;
#endif

typedef OWLGeometry OWLTriangles;

OWL_API OWLContext owlContextCreate();
OWL_API void owlContextDestroy(OWLContext context);
OWL_API OWLModule  owlContextCreateModule(const char *ptxCode);
OWL_API OWLGeometry  owlContextCreateGeometry(OWLContext context,
                                              OWLGeometryType type);
OWL_API OWLGeometryGroup  owlContextCreateGeometryGroup(OWLContext context,
                                                        size_t numGeometries,
                                                        OWLGeometry *initValues);
OWL_API OWLInstanceGroup  owlContextCreateInstanceGroup(OWLContext context,
                                                        size_t numInstances);

OWL_API OWLGeometryType owlContextCreateGeometryType(OWLContext context,
                                                     OWLGeometryKind kind,
                                                     size_t sizeOfArgs);
OWL_API OWLBuffer owlBufferCreate(OWLContext context,
                                  OWLDataType type,
                                  int num,
                                  const void *init);
// OWL_API OWLTriangles owlTrianglesCreate(OWLContext context,
//                                         size_t varsStructSize);
OWL_API void owlTrianglesSetVertices(OWLTriangles triangles,
                                     OWLBuffer vertices);
OWL_API void owlTrianglesSetIndices(OWLTriangles triangles,
                                     OWLBuffer indices);

OWL_API void owlGeometryTypeDeclareVariable(OWLGeometryType object,
                                         const char *varName,
                                         OWLDataType type,
                                         size_t offset);
OWL_API OWLVariable owlGeometryGetVariable(OWLGeometry geom,
                                           const char *varName);

OWL_API void owlObjectRelease(OWLObject object);
OWL_API void owlTrianglesRelease(OWLTriangles triangles);
OWL_API void owlVariableRelease(OWLVariable variable);
OWL_API void owlBufferRelease(OWLBuffer buffer);


OWL_API void owlGeometryTypeSetClosestHitProgram(OWLGeometryType type,
                                                 int rayType,
                                                 OWLModule module,
                                                 const char *progName);

// -------------------------------------------------------
// variable setters
// -------------------------------------------------------
OWL_API void owlVariableSet1f(OWLVariable variable, const float value);
OWL_API void owlVariableSet3fv(OWLVariable variable, const float *value);

#define OWL_ALL_RAY_TYPES -1
