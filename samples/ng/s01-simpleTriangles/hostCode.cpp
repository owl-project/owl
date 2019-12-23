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

const char *outFileName = "ll01-simpleTriangles.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  // owl::ll::DeviceGroup::SP ll
  //   = owl::ll::DeviceGroup::create();
  OWLContext context = owlContextCreate();
  
  LOG("building pipeline ...");

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  // ll->allocModules(1);
  // ll->setModule(0,ptxCode);
  // ll->buildModules();
  OWLModule module = owlModuleCreate(context,ptxCode);
  
  // enum { TRIANGLES_GEOM_TYPE=0,NUM_GEOM_TYPES };
  // ll->allocGeomTypes(NUM_GEOM_TYPES);

  OWLVarDecl trianglesGeomVars[]
    = {
       { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
       { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
       { "color",  OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)}
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,3);
  
  // ll->setGeomTypeClosestHit(/*program ID*/TRIANGLES_GEOM_TYPE,
  //                           /*ray type  */0,
  //                           /*module:*/0,
  //                           "TriangleMesh");
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");
  // ll->allocRayGens(1);
  // ll->setRayGen(/*program ID*/0,
  //               /*module:*/0,
  //               "simpleRayGen");  
  OWLVarDecl rayGenVars[]
    = {
       { "deviceIndex",   OWL_INT,    OWL_OFFSETOF(RayGenData,deviceIndex)},
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
  // ll->allocMissProgs(1);
  // ll->setMissProg(/*program ID*/0,
  //                 /*module:*/0,
  //                 "miss");
  OWLVarDecl missProgVars[]
    = {
       { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
       { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
       { /* sentinel to mark end of list */ }
  };
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});
  
  // ll->buildPrograms();
  // ll->createPipeline();
  owlBuildPrograms(context);
  owlBuildPipeline(context);

  LOG("building geometries ...");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  
  // enum { VERTEX_BUFFER=0,INDEX_BUFFER,FRAME_BUFFER,NUM_BUFFERS };
  // ll->reallocBuffers(NUM_BUFFERS);
  // ll->createDeviceBuffer(VERTEX_BUFFER,NUM_VERTICES,sizeof(vec3f),vertices);
  // ll->createDeviceBuffer(INDEX_BUFFER,NUM_INDICES,sizeof(vec3i),indices);
  // ll->createHostPinnedBuffer(FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));

  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,NUM_VERTICES,vertices);
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,NUM_INDICES,indices);
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  
  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  // enum { TRIANGLES_GEOM=0,NUM_GEOMS };
  // ll->reallocGeoms(NUM_GEOMS);
  // ll->createTrianglesGeom(/* geom ID    */TRIANGLES_GEOM,
  //                         /* type/PG ID */TRIANGLES_GEOM_TYPE);
  // ll->trianglesGeomSetVertexBuffer(/* geom ID   */TRIANGLES_GEOM,
  //                                  /* buffer ID */VERTEX_BUFFER,
  //                                  /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  // ll->trianglesGeomSetIndexBuffer(/* geom ID   */TRIANGLES_GEOM,
  //                                 /* buffer ID */INDEX_BUFFER,
  //                                 /* meta info */NUM_INDICES,sizeof(vec3i),0);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_INDICES,sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  owlGeomSet3f(trianglesGeom,"color",owl3f{0,1,0});
  
  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  // enum { TRIANGLES_GROUP=0,NUM_GROUPS };
  // ll->reallocGroups(NUM_GROUPS);
  // int geomsInGroup[] = { 0 };
  // ll->createTrianglesGeomGroup(/* group ID */TRIANGLES_GROUP,
  //                              /* geoms in group, pointer */ geomsInGroup,
  //                              /* geoms in group, count   */ 1);
  // ll->groupBuildAccel(TRIANGLES_GROUP);

  OWLGroup trianglesGroup
    = owlTrianglesGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);







  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

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

  // TODO: FIX THIS
  std::cout << GDT_TERMINAL_RED << "WARNING: NOT CORRECTLY SETTING DEVICE INDEX AND COUNT YET" << GDT_TERMINAL_DEFAULT << std::endl;
  owlRayGenSet1i    (rayGen,"deviceIndex",  0);
  owlRayGenSet1i    (rayGen,"deviceCount",  1);
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSetBuffer(rayGen,"fbSize",       frameBuffer);
  owlRayGenSetGroup (rayGen,"world",        trianglesGroup);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  

  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // // ----------- build hitgroups -----------
  // const size_t maxHitGroupDataSize = sizeof(TriangleGroupData);
  // ll->sbtGeomTypesBuild
  //   (maxHitGroupDataSize,
  //    [&](uint8_t *output,int devID,int geomID,int rayID) {
  //     TriangleGroupData &self = *(TriangleGroupData*)output;
  //     self.color  = vec3f(0,1,0);
  //     self.index  = (vec3i*)ll->bufferGetPointer(INDEX_BUFFER,devID);
  //     self.vertex = (vec3f*)ll->bufferGetPointer(VERTEX_BUFFER,devID);
  //   });
  
  // // ----------- build miss prog(s) -----------
  // const size_t maxMissProgDataSize = sizeof(MissProgData);
  // ll->sbtMissProgsBuild
  //   (maxMissProgDataSize,
  //    [&](uint8_t *output,
  //        int devID,
  //        int rayType) {
  //     /* we don't have any ... */
  //     ((MissProgData*)output)->color0 = vec3f(.8f,0.f,0.f);
  //     ((MissProgData*)output)->color1 = vec3f(.8f,.8f,.8f);
  //   });
  
  // // ----------- build raygens -----------
  // const size_t maxRayGenDataSize = sizeof(RayGenData);
  // ll->sbtRayGensBuild
  //   (maxRayGenDataSize,
  //    [&](uint8_t *output,
  //        int devID,
  //        int rgID) {
  //     RayGenData *rg = (RayGenData*)output;
  //     rg->deviceIndex   = devID;
  //     rg->deviceCount = ll->getDeviceCount();
  //     rg->fbSize = fbSize;
  //     rg->fbPtr  = (uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,devID);
  //     rg->world  = ll->groupGetTraversable(TRIANGLES_GROUP,devID);

  //     // compute camera frame:
  //     vec3f &pos = rg->camera.pos;
  //     vec3f &d00 = rg->camera.dir_00;
  //     vec3f &ddu = rg->camera.dir_du;
  //     vec3f &ddv = rg->camera.dir_dv;
  //     float aspect = fbSize.x / float(fbSize.y);
  //     pos = lookFrom;
  //     d00 = normalize(lookAt-lookFrom);
  //     ddu = cosFovy * aspect * normalize(cross(d00,lookUp));
  //     ddv = cosFovy * normalize(cross(ddu,d00));
  //     d00 -= 0.5f * ddu;
  //     d00 -= 0.5f * ddv;
  //   });
  // LOG_OK("everything set up ...");


  owlBuildSBT(context);

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("trying to launch ...");
  // ll->launch(0,fbSize);
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  // const uint32_t *fb = (const uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,0);
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
  // owl::ll::DeviceGroup::destroy(ll);
  owlContextDestroy(context);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
