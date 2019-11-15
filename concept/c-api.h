
#ifdef __cplusplus
# define OWL_API extern "C"
#else
# define OWL_API /* bla */
#endif

typedef enum {
  OWL_FLOAT3,
  OWL_INT3
} OWLDataType;

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
OWL_API OWLTriangles owlTrianglesCreate(OWLContext context);
OWL_API void owlTrianglesSetVertices(OWLTriangles triangles,
                                     OWLBuffer vertices);
OWL_API void owlTrianglesSetIndices(OWLTriangles triangles,
                                     OWLBuffer indices);

OWL_API OWLVariable owlTrianglesGetVariable(OWLTriangles triangles,
                                            const char *varName);
OWL_API OWLVariable owlContextGetVariable(OWLContext context,
                                          const char *varName);


OWL_API void owlObjectRelease(OWLObject object);
OWL_API void owlTrianglesRelease(OWLTriangles triangles);
OWL_API void owlVariableRelease(OWLVariable variable);
OWL_API void owlBufferRelease(OWLBuffer buffer);

