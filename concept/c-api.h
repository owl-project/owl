
#pragma once

#include <sys/types.h>
#include <stdint.h>

#ifdef __cplusplus
# include <cstddef> // is this c++??
#endif

#define OWL_OFFSETOF(type,member)               \
  ((char *)(&((struct type *)0)-> member )             \
   -                                            \
   (char *)(((struct type *)0)))

#ifdef __cplusplus
# define OWL_API extern "C"
#else
# define OWL_API /* bla */
#endif

typedef enum {
  OWL_FLOAT3,
  OWL_INT3,
  OWL_BUFFER,
  OWL_BUFFER_SIZE,
  OWL_BUFFER_POINTER
} OWLDataType;

typedef struct _OWL_int3   { int32_t x,y,z; } OWL_int3;
typedef struct _OWL_float3 { float   x,y,z; } OWL_float3;

#ifdef __cplusplus
typedef struct _OWLObject {}   *OWLObject;
typedef struct _OWLContext   : public _OWLObject {} *OWLContext;
typedef struct _OWLBuffer    : public _OWLObject {} *OWLBuffer;
typedef struct _OWLTriangles : public _OWLObject {} *OWLTriangles;
typedef struct _OWLVariable  : public _OWLObject {} *OWLVariable;
#else
typedef struct _OWLObject    *OWLObject;
typedef struct _OWLContext   *OWLContext;
typedef struct _OWLBuffer    *OWLBuffer;
typedef struct _OWLTriangles *OWLTriangles;
typedef struct _OWLVariable  *OWLVariable;
#endif


OWL_API OWLContext owlContextCreate();
OWL_API void owlContextDestroy(OWLContext context);

OWL_API OWLBuffer owlBufferCreate(OWLContext context,
                                  OWLDataType type,
                                  int num,
                                  const void *init);
OWL_API OWLTriangles owlTrianglesCreate(OWLContext context,
                                        size_t varsStructSize);
OWL_API void owlTrianglesSetVertices(OWLTriangles triangles,
                                     OWLBuffer vertices);
OWL_API void owlTrianglesSetIndices(OWLTriangles triangles,
                                     OWLBuffer indices);

OWL_API void owlTrianglesDeclareVariable(OWLTriangles object,
                                         const char *varName,
                                         OWLDataType type,
                                         size_t offset);
OWL_API OWLVariable owlTrianglesGetVariable(OWLTriangles triangles,
                                            const char *varName);

OWL_API void owlObjectRelease(OWLObject object);
OWL_API void owlTrianglesRelease(OWLTriangles triangles);
OWL_API void owlVariableRelease(OWLVariable variable);
OWL_API void owlBufferRelease(OWLBuffer buffer);


// -------------------------------------------------------
// variable setters
// -------------------------------------------------------
OWL_API void owlVariableSet3fv(OWLVariable variable, const float *value);

