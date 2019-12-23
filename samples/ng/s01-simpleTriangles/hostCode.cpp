// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
//                                                                          //
// Licensed under the Apache License, Version 2.0 (the "License");          //
// you may not use this file except in compliance with the License.         //
// You may obtain a copy of the License at                                  //
//                                                                          //
//     http://www.apache.org/licenses/LICENSE-2.0                           //
//                                                                          //
// Unless required by applicable law or agreed to in writing, software      //
// distributed under the License is distributed on an "AS IS" BASIS,        //
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. //
// See the License for the specific language governing permissions and      //
// limitations under the License.                                           //
// ======================================================================== //

#include "owl/owl.h"
#include "deviceCode.h"




// -------------------------------------------------------
// VariableSet for different *object* types
// -------------------------------------------------------
struct owl2i { int x,y; };
struct owl3f { float x,y,z; };

inline void owlRayGenSetGroup(OWLRayGen rayGen, const char *varName, OWLGroup v)
{
  OWLVariable var = owlRayGenGetVariable(rayGen,varName);
  owlVariableSetGroup(var,v);
  owlVariableRelease(var);
}
inline void owlRayGenSetBuffer(OWLRayGen rayGen, const char *varName, OWLBuffer v)
{
  OWLVariable var = owlRayGenGetVariable(rayGen,varName);
  owlVariableSetBuffer(var,v);
  owlVariableRelease(var);
}
inline void owlGeomSetBuffer(OWLGeom rayGen, const char *varName, OWLBuffer v)
{
  OWLVariable var = owlGeomGetVariable(rayGen,varName);
  owlVariableSetBuffer(var,v);
  owlVariableRelease(var);
}


inline void owlRayGenSet1i(OWLRayGen rayGen, const char *varName, int v)
{
  OWLVariable var = owlRayGenGetVariable(rayGen,varName);
  owlVariableSet1i(var,v);
  owlVariableRelease(var);
}

inline void owlRayGenSet2i(OWLRayGen rayGen, const char *varName, const owl2i &v)
{
  OWLVariable var = owlRayGenGetVariable(rayGen,varName);
  owlVariableSet2iv(var,&v.x);
  owlVariableRelease(var);
}


inline void owlRayGenSet3f(OWLRayGen rayGen, const char *varName, const owl3f &v)
{
  OWLVariable var = owlRayGenGetVariable(rayGen,varName);
  owlVariableSet3fv(var,&v.x);
  owlVariableRelease(var);
}
inline void owlMissProgSet3f(OWLMissProg rayGen, const char *varName, const owl3f &v)
{
  OWLVariable var = owlMissProgGetVariable(rayGen,varName);
  owlVariableSet3fv(var,&v.x);
  owlVariableRelease(var);
}
inline void owlGeomSet3f(OWLGeom rayGen, const char *varName, const owl3f &v)
{
  OWLVariable var = owlGeomGetVariable(rayGen,varName);
  owlVariableSet3fv(var,&v.x);
  owlVariableRelease(var);
}






#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << GDT_TERMINAL_BLUE;                       \
  std::cout << "#owl.ng.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#owl.ng.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
  {
   { -1.f,-1.f,-1.f },
   { +1.f,-1.f,-1.f },
   { -1.f,+1.f,-1.f },
   { +1.f,+1.f,-1.f },
   { -1.f,-1.f,+1.f },
   { +1.f,-1.f,+1.f },
   { -1.f,+1.f,+1.f },
   { +1.f,+1.f,+1.f }
  };

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
  {
   { 0,1,3 }, { 2,3,0 },
   { 5,7,6 }, { 5,6,4 },
   { 0,4,5 }, { 0,5,1 },
   { 2,3,7 }, { 2,7,6 },
   { 1,5,7 }, { 1,7,3 },
   { 4,0,2 }, { 4,2,6 }
  };

const char *outFileName = "ng01-simpleTriangles.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  OWLContext context = owlContextCreate();
  OWLModule module = owlModuleCreate(context,ptxCode);
  
  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)}
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,3);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,NUM_VERTICES,vertices);
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,NUM_INDICES,indices);
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_INDICES,sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  owlGeomSet3f(trianglesGeom,"color",owl3f{0,1,0});
  
  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog 
  // -------------------------------------------------------

  OWLVarDecl missProgVars[]
    = {
       { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
       { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
       { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  
  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------

  OWLVarDecl rayGenVars[] = {
    { "deviceIndex",   OWL_DEVICE, OWL_OFFSETOF(RayGenData,deviceIndex)},
    { "deviceCount",   OWL_INT,    OWL_OFFSETOF(RayGenData,deviceCount)},
    { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { /* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

  // ----------- compute variable values  ------------------
  vec3f camera_pos = lookFrom;
  vec3f camera_d00
    = normalize(lookAt-lookFrom);
  float aspect = fbSize.x / float(fbSize.y);
  vec3f camera_ddu
    = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
  vec3f camera_ddv
    = cosFovy * normalize(cross(camera_ddu,camera_d00));
  camera_d00 -= 0.5f * camera_ddu;
  camera_d00 -= 0.5f * camera_ddv;

  // ----------- set variables  ----------------------------
  owlRayGenSet1i    (rayGen,"deviceCount",  owlGetDeviceCount(context));
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        trianglesGroup);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("launching ...");
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb
    = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  assert(fb);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  owlContextDestroy(context);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
