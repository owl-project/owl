// ======================================================================== //
// Copyright 2019-2025 Ingo Wald                                            //
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

// main primer API
#include <owl/owl_prime.h>
// common math/box stuff
#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"
#include "owl/common/math/AffineSpace.h"
#include <vector>
#include <map>
#include <set>
#include <thread>
#include <mutex>
#include <condition_variable>

namespace primer {
  using namespace owl::common;

  /*! helper function that allows for accesing strided arrays given by
      a base pointer and a strided */
  template<typename T>
  inline __both__ T &strided(T *basePtr, size_t stride, size_t arrayIndex)
  { return *((T*)(((uint8_t*)basePtr)+stride*arrayIndex)); }
    
#define PRIMER_NYI throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" function not yet implemented")
#define PRIMER_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" function not yet implemented")
  
  
  /*! our internal C++ representation of a ray - MUST have same data
   *  layout as OPRay in include/primer.h */
  struct Ray {
    vec3f origin;
    float tMin;
    vec3f direction;
    float tMax;
  };
  
  /*! our internal C++ representation of a hit - MUST have same data
   *  layout as OPHit in include/primer.h */
  struct Hit : public OPHit {
    /*! invalidate this hit record to "no hit found" */
    inline __both__ void clear() { instID = primID = -1; }
  };
  
}
