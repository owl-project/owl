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

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << GDT_TERMINAL_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
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
  LOG("ll example '" << av[0] << "' starting up");

  // owl::ll::DeviceGroup::SP ll
  //   = owl::ll::DeviceGroup::create();
  LLOContext llo = lloContextCreate(nullptr,0);

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");
  
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);

  enum { TRIANGLES_GEOM_TYPE=0,NUM_GEOM_TYPES };
  // ll->allocGeomTypes(NUM_GEOM_TYPES);
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  // ll->geomTypeCreate(TRIANGLES_GEOM_TYPE,sizeof(TrianglesGeomData));
  lloGeomTypeCreate(llo,TRIANGLES_GEOM_TYPE,sizeof(TrianglesGeomData));
  // ll->setGeomTypeClosestHit(/*program ID*/TRIANGLES_GEOM_TYPE,
  //                           /*ray type  */0,
  //                           /*module:*/0,
  //                           "TriangleMesh");
  lloGeomTypeClosestHit(llo,
                        /*program ID*/TRIANGLES_GEOM_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "TriangleMesh");
  
  lloAllocRayGens(llo,1);
  lloRayGenCreate(llo,
                  /*program ID*/0,
                  /*module:*/0,
                  "simpleRayGen",
                  sizeof(RayGenData));
  
  // ll->allocMissProgs(1);
  lloAllocMissProgs(llo,1);
  // ll->setMissProg(/*program ID*/0,
  //                 /*module:*/0,
  //                 "miss",
  //                 sizeof(MissProgData));
  lloMissProgCreate(llo,
                    /*program ID*/0,
                    /*module:*/0,
                    "miss",
                    sizeof(MissProgData));
  
  lloAllocMissProgs(llo,1);
  lloMissProgCreate(llo,
                    /*program ID*/0,
                    /*module:*/0,
                    "miss",
                    sizeof(MissProgData));

  // ll->buildPrograms();
  // ll->createPipeline();
  lloBuildPrograms(llo);
  lloCreatePipeline(llo);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { VERTEX_BUFFER=0,INDEX_BUFFER,FRAME_BUFFER,NUM_BUFFERS };
  // ll->allocBuffers(NUM_BUFFERS);
  lloAllocBuffers(llo,NUM_BUFFERS);
  // ll->createDeviceBuffer(VERTEX_BUFFER,NUM_VERTICES,sizeof(vec3f),vertices);
  lloDeviceBufferCreate(llo,VERTEX_BUFFER,NUM_VERTICES*sizeof(vec3f),vertices);
  // ll->createDeviceBuffer(INDEX_BUFFER,NUM_INDICES,sizeof(vec3i),indices);
  lloDeviceBufferCreate(llo,INDEX_BUFFER,NUM_INDICES*sizeof(vec3i),indices);
  // ll->createHostPinnedBuffer(FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));
  lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y*sizeof(uint32_t));
  
  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  enum { TRIANGLES_GEOM=0,NUM_GEOMS };
  // ll->allocGeoms(NUM_GEOMS);
  lloAllocGeoms(llo,NUM_GEOMS);
  // ll->trianglesGeomCreate(/* geom ID    */TRIANGLES_GEOM,
  //                         /* type/PG ID */TRIANGLES_GEOM_TYPE);
  lloTrianglesGeomCreate(llo,
                         /* geom ID    */TRIANGLES_GEOM,
                         /* type/PG ID */TRIANGLES_GEOM_TYPE);
  // ll->trianglesGeomSetVertexBuffer(/* geom ID   */TRIANGLES_GEOM,
  //                                  /* buffer ID */VERTEX_BUFFER,
  //                                  /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  lloTrianglesGeomSetVertexBuffer(llo,
                                  /* geom ID   */TRIANGLES_GEOM,
                                  /* buffer ID */VERTEX_BUFFER,
                                  /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  // ll->trianglesGeomSetIndexBuffer(/* geom ID   */TRIANGLES_GEOM,
  //                                 /* buffer ID */INDEX_BUFFER,
  //                                 /* meta info */NUM_INDICES,sizeof(vec3i),0);
  lloTrianglesGeomSetIndexBuffer(llo,
                                 /* geom ID   */TRIANGLES_GEOM,
                                 /* buffer ID */INDEX_BUFFER,
                                 /* meta info */NUM_INDICES,sizeof(vec3i),0);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  enum { TRIANGLES_GROUP=0,NUM_GROUPS };
  // ll->allocGroups(NUM_GROUPS);
  lloAllocGroups(llo,NUM_GROUPS);
  int geomsInGroup[] = { 0 };
  // ll->trianglesGeomGroupCreate(/* group ID */TRIANGLES_GROUP,
  //                              /* geoms in group, pointer */ geomsInGroup,
  //                              /* geoms in group, count   */ 1);
  lloTrianglesGeomGroupCreate(llo,
                              /* group ID */TRIANGLES_GROUP,
                              /* geoms in group, pointer */ geomsInGroup,
                              /* geoms in group, count   */ 1);
  // ll->groupBuildAccel(TRIANGLES_GROUP);
  lloGroupBuildAccel(llo,TRIANGLES_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  // ll->sbtHitProgsBuild
  //   (
  lloSbtHitProgsBuild
    (llo,
     [&](uint8_t *output,int devID,int geomID,int rayID) {
      TrianglesGeomData &self = *(TrianglesGeomData*)output;
      self.color  = vec3f(0,1,0);
      // self.index  = (vec3i*)ll->bufferGetPointer(INDEX_BUFFER,devID);
      self.index  = (vec3i*)lloBufferGetPointer(llo,INDEX_BUFFER,devID);
      // self.vertex = (vec3f*)ll->bufferGetPointer(VERTEX_BUFFER,devID);
      self.vertex = (vec3f*)lloBufferGetPointer(llo,VERTEX_BUFFER,devID);
    });
  
  // ----------- build miss prog(s) -----------
  // ll->sbtMissProgsBuild
  //   (
  lloSbtMissProgsBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rayType) {
      /* we don't have any ... */
      ((MissProgData*)output)->color0 = vec3f(.8f,0.f,0.f);
      ((MissProgData*)output)->color1 = vec3f(.8f,.8f,.8f);
    });
  
  // ----------- build raygens -----------
  // ll->sbtRayGensBuild
  //   ([&](uint8_t *output,
  //        int devID,
  //        int rgID) {
  lloSbtRayGensBuild
    (llo,[&](uint8_t *output,
             int devID,
             int rgID) {
      RayGenData *rg = (RayGenData*)output;
      rg->deviceIndex   = devID;
      // rg->deviceCount = ll->getDeviceCount();
      rg->deviceCount = lloGetDeviceCount(llo);
      rg->fbSize = fbSize;
      // rg->fbPtr  = (uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,devID);
      rg->fbPtr  = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
      // rg->world  = ll->groupGetTraversable(TRIANGLES_GROUP,devID);
      rg->world  = lloGroupGetTraversable(llo,TRIANGLES_GROUP,devID);

      // compute camera frame:
      vec3f &pos = rg->camera.pos;
      vec3f &d00 = rg->camera.dir_00;
      vec3f &ddu = rg->camera.dir_du;
      vec3f &ddv = rg->camera.dir_dv;
      float aspect = fbSize.x / float(fbSize.y);
      pos = lookFrom;
      d00 = normalize(lookAt-lookFrom);
      ddu = cosFovy * aspect * normalize(cross(d00,lookUp));
      ddv = cosFovy * normalize(cross(ddu,d00));
      d00 -= 0.5f * ddu;
      d00 -= 0.5f * ddv;
    });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  // ll->launch(0,fbSize);
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  // const uint32_t *fb = (const uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,0);
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying llo context ...");
  // owl::ll::DeviceGroup::destroy(ll);
  lloContextDestroy(llo);

  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
