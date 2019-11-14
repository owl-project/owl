
#ifdef __cplusplus
# define OWL_API extern "C"
#else
# define OWL_API /* bla */
#endif

typedef enum {
  OWL_FLOAT3,
  OWL_INT3
} OWLDataType;
  
typedef struct _OWLContext   *OWLContext;
typedef struct _OWLBuffer    *OWLBuffer;
typedef struct _OWLTriangles *OWLTriangles;
typedef struct _OWLVariable  *OWLVariable;

OWL_API OWLContext owlContextCreate();
OWL_API void owlContextDestroy(OWLContext context);

OWL_API OWLBuffer owlCreateBuffer(OWLContext context,
                                  OWLDataType type,
                                  int num,
                                  const void *init);
OWL_API OWLTriangles owlTrianglesCreate(OWLContext context);
OWL_API void owlTrianglesSetVertices(OWLContext context,
                                     OWLBuffer buffer);
OWL_API void owlTrianglesSetVertices(OWLContext context,
                                     OWLBuffer buffer);
OWL_API void owlTrianglesRelease(OWLVariable variable);

OWL_API void owlTrianglesGetVariable(OWLTriangles triangles,
                                     const char *varName);
OWL_API void owlVariableRelease(OWLVariable variable);

