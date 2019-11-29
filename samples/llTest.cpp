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

extern char ptxCode[1];

gdt::vec3f unitCube_vertices[8] =
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

gdt::vec3i unitCube_indices[] =
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
  ll->reallocGeometries(1);
  ll->createGeometryTriangles(/* geom ID    */0,
                              /* type/PG ID */0,
                              /* primcount  */1);
  
  ll->reallocGroups(1);

  std::cout << GDT_TERMINAL_BLUE;
  std::cout << "#######################################################" << std::endl;
  std::cout << "actual ll-work here ..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;
}
