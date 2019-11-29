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

#include "../owl/ll/Device.h"

using gdt::vec3f;
using gdt::vec3i;

extern char ptxCode[1];

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
  {
   { 0.f,0.f,0.f },
   { 1.f,0.f,0.f },
   { 0.f,1.f,0.f },
   { 1.f,1.f,0.f },
   { 0.f,0.f,1.f },
   { 1.f,0.f,1.f },
   { 0.f,1.f,1.f },
   { 1.f,1.f,1.f }
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


int main(int ac, char **av)
{
  owl::ll::DeviceGroup::SP ll
    = owl::ll::DeviceGroup::create();

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
  
  ll->allocRayGenPGs(1);
  ll->setRayGenPG(/*program ID*/0,
                  /*module:*/0,
                  "simpleRayGen");
  
  ll->allocMissPGs(1);
  ll->setMissPG(/*program ID*/0,
                /*module:*/0,
                "defaultRayType");
  ll->buildPrograms();
  ll->createPipeline();


  // ##################################################################
  // set up all the *GEOMETRIES* we want to run that code on
  // ##################################################################

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { VERTEX_BUFFER=0,INDEX_BUFFER,NUM_BUFFERS };
  ll->reallocBuffers(NUM_BUFFERS);
  ll->createDeviceBuffer(VERTEX_BUFFER,NUM_VERTICES,sizeof(vec3f),vertices);
  ll->createDeviceBuffer(INDEX_BUFFER,NUM_INDICES,sizeof(vec3i),vertices);
  
  // ------------------------------------------------------------------
  // alloc geometry
  // ------------------------------------------------------------------
  ll->reallocGeometries(1);
  ll->createTrianglesGeometry(/* geom ID    */0,
                              /* type/PG ID */0);
  ll->triangleGeometrySetVertices(/* geom ID     */ 0,
                                  /* buffer ID */0,
                                  /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  ll->triangleGeometrySetIndices(/* geom ID     */ 0,
                                 /* buffer ID */1,
                                 /* meta info */sizeof(vec3i),0);
  
  ll->reallocGroups(1);
  int geomsInGroup[] = { 0 };
  ll->createTrianglesGeometryGroup(/* group ID */0,
                                   /* geoms in group, pointer */ geomsInGroup,
                                   /* geoms in group, count   */ 1);
                           

  std::cout << GDT_TERMINAL_BLUE;
  std::cout << "#######################################################" << std::endl;
  std::cout << "actual ll-work here ..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;
}
