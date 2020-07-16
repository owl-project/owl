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

#include "Context.h"

namespace owl {
  
  void DeviceContext::destroyPipeline()
  {
    if (!device->context->pipeline) return;
    
    SetActiveGPU forLifeTime(device);
    
    OPTIX_CHECK(optixPipelineDestroy(device->context->pipeline));
    device->context->pipeline = 0;
    
    // device->popActive(oldActive);
  }
  
  void DeviceContext::buildPipeline()
  {
    SetActiveGPU forLifeTime(device);
    
    auto &allPGs = allActivePrograms;
    if (allPGs.empty())
      throw std::runtime_error("trying to create a pipeline w/ 0 programs!?");
      
    char log[2048];
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK(optixPipelineCreate(device->context->optixContext,
                                    &device->context->pipelineCompileOptions,
                                    &device->context->pipelineLinkOptions,
                                    allPGs.data(),
                                    (uint32_t)allPGs.size(),
                                    log,&sizeof_log,
                                    &device->context->pipeline
                                    ));
      
    uint32_t maxAllowedByOptix = 0;
    optixDeviceContextGetProperty
      (device->context->optixContext,
       OPTIX_DEVICE_PROPERTY_LIMIT_MAX_TRAVERSABLE_GRAPH_DEPTH,
       &maxAllowedByOptix,
       sizeof(maxAllowedByOptix));
    if (uint32_t(parent->maxInstancingDepth+1) > maxAllowedByOptix)
      throw std::runtime_error
        ("error when building pipeline: "
         "attempting to set max instancing depth to "
         "value that exceeds OptiX's MAX_TRAVERSABLE_GRAPH_DEPTH limit");

    PRINT(parent->maxInstancingDepth);
    PRINT(maxAllowedByOptix);
    
    OPTIX_CHECK(optixPipelineSetStackSize
                (device->context->pipeline,
                 /* [in] The pipeline to configure the stack size for */
                 2*1024,
                 /* [in] The direct stack size requirement for
                    direct callables invoked from IS or AH. */
                 2*1024,
                 /* [in] The direct stack size requirement for
                    direct callables invoked from RG, MS, or CH.  */
                 2*1024,
                 /* [in] The continuation stack requirement. */
                 int(parent->maxInstancingDepth+1)
                 /* [in] The maximum depth of a traversable graph
                    passed to trace. */
                 ));

    // device->popActive(oldActive);
  }

}
