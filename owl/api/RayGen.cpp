// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "RayGen.h"
#include "Context.h"

namespace owl {


/*! C++-only wrapper of callback method with lambda function */
template<typename Lambda>
void lloParamsLaunch2D(ll::DeviceGroup *llo,
                       int32_t      rayGenID,
                       int32_t      Nx,
                       int32_t      Ny,
                       int32_t      launchParamsObjectID,
                       const Lambda &l)
{
  llo->launch
    (rayGenID,vec2i(Nx,Ny),
     launchParamsObjectID,
     [](uint8_t *output,
        int devID,
        const void *cbData)
     {
       const Lambda *lambda = (const Lambda *)cbData;
       (*lambda)(output,devID);
     },
     (const void *)&l);
}


  
  RayGenType::RayGenType(Context *const context,
                         Module::SP module,
                         const std::string &progName,
                         size_t varStructSize,
                         const std::vector<OWLVarDecl> &varDecls)
    : SBTObjectType(context,context->rayGenTypes,varStructSize,varDecls),
      module(module),
      progName(progName)
  {
  }
  
  RayGen::RayGen(Context *const context,
                 RayGenType::SP type) 
    : SBTObject(context,context->rayGens,type)
  {
    assert(context);
    assert(type);
    assert(type.get());
    assert(type->module);
    assert(type->progName != "");
    context->llo->setRayGen(this->ID,
                    type->module->ID,
                    type->progName.c_str(),
                    type->varStructSize);
  }

  void RayGen::launch(const vec2i &dims)
  {
    context->llo->launch(this->ID,dims);
  }

  void RayGen::launch(const vec2i &dims, const LaunchParams::SP &lp)
  {
    lloParamsLaunch2D(context->llo,this->ID,dims.x,dims.y,lp->ID,
                      [&](uint8_t *launchParamsToWrite, int deviceID){
                        lp->writeVariables(launchParamsToWrite,deviceID);
                      });
                       // int32_t      rayGenID,
                       // int32_t      Nx,
                       // int32_t      Ny,
                       // int32_t      launchParamsObjectID,
                       // const Lambda &l)

  // auto lambda = [&](uint8_t *launchParamsToWrite, int deviceID){
    //                        lp->writeVariables(launchParamsToWrite,deviceID);
    //               };
    // context->llo->launch(this->ID,dims,lp->ID,
    //                      ,(const void *)&lp);
  }
  
} // ::owl

