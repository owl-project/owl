#include "c-api.h"
#include <stdio.h>
#include <stdlib.h>

struct TrianglesVars {
  OWL_int3   *index;
  OWL_float3 *vertex;
  OWL_float3  color;
};

int main(int ac, char **av)
{
  OWLContext  context  = owlContextCreate();
  float vertex[][3] = {
    { 0,0,0 },
    { 1,0,0 },
    { 0,1,0 }
  };
  int index[][3] = {
    { 0,1,2 }
  };
  float green[3] = { 0.f, 1.f, 0.f };
  OWLBuffer    vertices  = owlBufferCreate(context,OWL_FLOAT3,3,vertex);
  OWLBuffer    indices   = owlBufferCreate(context,OWL_INT3,1,index);
  OWLTriangles triangles = owlTrianglesCreate(context,sizeof(struct TrianglesVars));
  /* OWLTriangles triangles = owlTrianglesCreate(context,sizeof(struct TrianglesVars)); */
  owlTrianglesDeclareVariable(triangles,"vertex",OWL_BUFFER_POINTER,
                              OWL_OFFSETOF(TrianglesVars,vertex));
  owlTrianglesDeclareVariable(triangles,"index", OWL_BUFFER_POINTER,
                              OWL_OFFSETOF(TrianglesVars,index));
  owlTrianglesDeclareVariable(triangles,"diffuseColor",OWL_FLOAT3,
                              OWL_OFFSETOF(TrianglesVars,color));
  
  owlTrianglesSetVertices(triangles,vertices);
  owlTrianglesSetIndices(triangles,indices);
  
  OWLVariable diffuseColor = owlTrianglesGetVariable(triangles,"diffuseColor");
  owlVariableSet3fv(diffuseColor,green);
  
  owlBufferRelease(vertices);
  owlBufferRelease(indices);
  owlVariableRelease(diffuseColor);
  owlTrianglesRelease(triangles);
  owlContextDestroy(context);
}

