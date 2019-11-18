#include "c-api.h"
#include <stdio.h>
#include <stdlib.h>

struct TrianglesVars {
  OWL_int3   *index;
  OWL_float3 *vertex;
  OWL_float3  color;
};

struct RenderFrameArgs {
  OWLDeviceTraversable world;
  OWLDeviceBuffer2D    frameBuffer;
  OWL_float3 camera_org;
  OWL_float3 camera_dir_00;
  OWL_float3 camera_dir_dx;
  OWL_float3 camera_dir_dy;
  OWL_float3 bgColor;
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
  int imgSize[2] = { 800,600 };
  OWLBuffer    frameBuffer = owlBufferCreate(context,OWL_FLOAT3,3,vertex);
  
  float green[3] = { 0.f, 1.f, 0.f };
  OWLBuffer    vertices  = owlBufferCreate(context,OWL_FLOAT3,3,vertex);
  OWLBuffer    indices   = owlBufferCreate(context,OWL_INT3,1,index);

  OWLGeometryType diffuseTrianglesType
    = ownGeometryTypeCreate(OWL_TRIANGLES,sizeof(struct TrianglesVars));
  owlGeometryTypeDeclareVariable(diffuseTrianglesType,"vertex",OWL_BUFFER_POINTER,
                                 OWL_OFFSETOF(TrianglesVars,vertex));
  owlGeometryTypeDeclareVariable(diffuseTrianglesType,"index", OWL_BUFFER_POINTER,
                                 OWL_OFFSETOF(TrianglesVars,index));
  owlGeometryTypeDeclareVariable(diffuseTrianglesType,"diffuseColor",OWL_FLOAT3,
                                 OWL_OFFSETOF(TrianglesVars,color));
  
  OWLGeometry triangles = owlGeometryCreate(context,diffuseTriangleType);

  owlTrianglesSetVertices(triangles,vertices);
  owlBufferRelease(vertices);

  owlTrianglesSetIndices(triangles,indices);
  owlBufferRelease(indices);
  
  OWLVariable diffuseColor = owlTrianglesGetVariable(triangles,"diffuseColor");
  owlVariableSet3fv(diffuseColor,green);
  owlVariableRelease(diffuseColor);


  // ==================================================================
  // create and configure launch proram
  // ==================================================================
  OWLLaunchProg renderFrame
    = owlContextCreateLaunchProg(context,sizeof(RenderFrameArgs));
  {
  }
  
  // ==================================================================
  // launch renderframe program
  // ==================================================================
  owlContextLaunch2D(renderFrame,imgSize[0],imgSize[1]);
  
  owlGeometryRelease(triangles);
  owlContextDestroy(context);
}

