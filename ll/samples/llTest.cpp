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

#include "ll/DeviceGroup.h"
#include "deviceCode.h"

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

const char *outFileName = "llTest.png";
const vec2i fbSize(800,600);

int main(int ac, char **av)
{
  std::cout << GDT_TERMINAL_BLUE;
  std::cout << "###########################################################" << std::endl;
  std::cout << "llTest: mini-app for testing low-level optix wrapper api..." << std::endl;
  std::cout << "###########################################################" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;

  owl::ll::DeviceGroup::SP ll
    = owl::ll::DeviceGroup::create();

  LOG("building pipeline ...");
  std::cout << GDT_TERMINAL_DEFAULT;

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  ll->allocModules(1);
  ll->setModule(0,ptxCode);
  ll->buildModules();
  
  ll->allocHitGroupPGs(1);
  ll->setHitGroupClosestHit(/*program ID*/0,
                            /*module:*/0,
                            "TriangleMesh");
  
  ll->allocRayGens(1);
  ll->setRayGen(/*program ID*/0,
                /*module:*/0,
                "simpleRayGen");
  
  ll->allocMissProgs(1);
  ll->setMissProg(/*program ID*/0,
                  /*module:*/0,
                  "defaultRayType");
  ll->buildPrograms();
  ll->createPipeline();

  LOG("building geometries ...");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { VERTEX_BUFFER=0,INDEX_BUFFER,FRAME_BUFFER,NUM_BUFFERS };
  ll->reallocBuffers(NUM_BUFFERS);
  ll->createDeviceBuffer(VERTEX_BUFFER,NUM_VERTICES,sizeof(vec3f),vertices);
  ll->createDeviceBuffer(INDEX_BUFFER,NUM_INDICES,sizeof(vec3i),indices);
  ll->createHostPinnedBuffer(FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));
  
  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  ll->reallocGeoms(1);
  ll->createTrianglesGeom(/* geom ID    */0,
                          /* type/PG ID */0);
  ll->trianglesGeomSetVertexBuffer(/* geom ID     */ 0,
                                   /* buffer ID */VERTEX_BUFFER,
                                   /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  ll->trianglesGeomSetIndexBuffer(/* geom ID     */ 0,
                                  /* buffer ID */INDEX_BUFFER,
                                  /* meta info */NUM_INDICES,sizeof(vec3i),0);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  enum { TRIANGLES_GROUP=0,NUM_GROUPS };
  ll->reallocGroups(NUM_GROUPS);
  int geomsInGroup[] = { 0 };
  ll->createTrianglesGeomGroup(/* group ID */TRIANGLES_GROUP,
                               /* geoms in group, pointer */ geomsInGroup,
                               /* geoms in group, count   */ 1);
  ll->groupBuildAccel(TRIANGLES_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  const size_t maxHitGroupDataSize = sizeof(TriangleGroupData);
  ll->sbtHitGroupsBuild(maxHitGroupDataSize,
                        [&](uint8_t *output,
                            int devID,
                            int geomID,
                            int rayID,
                            const void *cbData) {
                          ((TriangleGroupData*)output)->color = vec3f(0,1,0);
                        });
  
  // ----------- build miss prog(s) -----------
  const size_t maxMissProgDataSize = sizeof(MissProgData);
  ll->sbtMissProgsBuild(maxMissProgDataSize,
                        [&](uint8_t *output,
                            int devID,
                            int rayType, 
                            const void *cbData) {
                          /* we don't have any ... */
                        });
  
  // ----------- build raygens -----------
  const size_t maxRayGenDataSize = sizeof(RayGenData);
  ll->sbtRayGensBuild(maxRayGenDataSize,
                      [&](uint8_t *output,
                          int devID,
                          int rgID, 
                          const void *cbData) {
                        RayGenData *rg = (RayGenData*)output;
                        rg->deviceIndex   = devID;
                        rg->deviceCount = ll->getDeviceCount();
                        rg->color0 = vec3f(.8,.8,.8);
                        rg->color1 = vec3f(.7,0,0);
                        rg->fbSize = fbSize;
                        rg->fbPtr  = (uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,devID);
                        rg->world  = ll->groupGetTraversable(TRIANGLES_GROUP,devID);
                      });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("trying to launch ...");
  ll->launch(0,fbSize);
  // todo: explicit sync?
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb = (const uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  owl::ll::DeviceGroup::destroy(ll);
  
  LOG_OK("seems all wen't ok; app is done, this should be the last output ...");
}
