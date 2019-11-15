#include "c-api.h"
#include <stdio.h>
#include <stdlib.h>

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
  OWLBuffer    vertices = owlBufferCreate(context,OWL_FLOAT3,3,vertex);
  OWLBuffer    indices  = owlBufferCreate(context,OWL_INT3,1,index);
  OWLTriangles triangles = owlTrianglesCreate(context);
  owlTrianglesSetVertices(triangles,vertices);
  owlTrianglesSetIndices(triangles,indices);
  
  OWLVariable context_world = owlContextGetVariable(context,"world");
  owlBufferRelease(vertices);
  owlBufferRelease(indices);
  owlVariableRelease(context_world);
  owlTrianglesRelease(triangles);
  owlContextDestroy(context);
}

