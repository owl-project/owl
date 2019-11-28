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

int main(int ac, char **av)
{
  owl::ll::Devices::SP ll
    = owl::ll::Devices::create();

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

  std::cout << GDT_TERMINAL_BLUE;
  std::cout << "#######################################################" << std::endl;
  std::cout << "actual ll-work here ..." << std::endl;
  std::cout << "#######################################################" << std::endl;
  std::cout << GDT_TERMINAL_DEFAULT;
}
