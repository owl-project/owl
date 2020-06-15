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

#pragma once

#include "owl_host.h"
#include <iostream>

namespace owl {
  struct DeprecatedWarning {
    DeprecatedWarning(const char *old, const char *newName)
    {
      static bool warned = false;
      if (warned) return;
      std::cout << "@@ OWL DEPRECATION WARNING: function '" << old
                << "' is now called '" << newName << "'; "
                << "the old naming is still supported but will soon be removed"
                << std::endl;
      warned = true;
    }
  };
}

/*! executes an optix lauch of given size, with given launch
  program. Note this call is asynchronous, and may _not_ be
  completed by the time this function returns. */
inline void
owlParamsLaunch2D(OWLRayGen rayGen, int dims_x, int dims_y,
                  OWLLaunchParams lp)
{
  static owl::DeprecatedWarning deprecated("owlParamsLaunch2D","owlLaunch2D");
  owlLaunch2D(rayGen,dims_x,dims_y,lp);
}

inline OWLLaunchParams
owlLaunchParamsCreate(OWLContext  context,
                      size_t      sizeOfVarStruct,
                      OWLVarDecl *vars,
                      size_t      numVars)
{
  static owl::DeprecatedWarning deprecated("owlParamsLaunch2D","owlLaunch2D");
  return owlParamsCreate(context,sizeOfVarStruct,vars,numVars);
}



