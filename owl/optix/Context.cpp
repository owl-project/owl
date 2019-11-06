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

#include "optix/Context.h"

namespace optix {

  /*! creates a new context with the given device IDs. Invalid
    device IDs get ignored with a warning, but if no device can be
    created at all an error will be thrown */
  Context::SP Context::create(const std::vector<int> &deviceIDs)
  {
    OWL_NOTIMPLEMENTED;
  }

  /*! creates a new context with the given device IDs. Invalid
    device IDs get ignored with a warning, but if no device can be
    created at all an error will be thrown */
  Context::SP Context::create(GPUSelectionMethod whichGPUs)
  {
    switch(whichGPUs) {
    default:
      throw optix::Error("Context::create()",
                         "specified GPU Selection method not implemented "
                         "(method="+std::to_string((int)whichGPUs)+")"
                         );
    }
  }
  
} // ::optix
