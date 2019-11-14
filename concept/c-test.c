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
  OWLBuffer    vertices = owlCreateBuffer(context,OWL_FLOAT3,3,vertex);
  OWLBuffer    indices  = owlCreateBuffer(context,OWL_INT3,1,index);
  OWLTriangles geometry = owlCreateTriangles(context);
  owlTrianglesSetVertices(geometry,vertices);
  owlTrianglesSetIndices(geometry,indices);
  
  OWLVariable context_world = owlContextGetVariable(context,"world");
  owlBufferRelease(vertices);
  owlBufferRelease(indices);
  owlVariableRelease(context_world);
  owlGeometryRelease(geometry);
  owlContextDestroy(context);
}
