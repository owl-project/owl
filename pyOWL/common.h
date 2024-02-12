// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <map>
#include <vector>
#include <stdint.h>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

namespace pyOWL {

  namespace py = pybind11;
  
  using namespace owl;
  using namespace owl::common;

  /*! convert pybind tuple to owl vector type (w/ size sanity check) */
  inline vec2i make_vec2i(const std::vector<int> &vec)
  {
    if (vec.size() != 2)
      throw std::runtime_error("the tuple passed didn't have the"
                               " expected number of elements (2)");
    return vec2i(vec[0],vec[1]);
  }
  
  /*! convert pybind tuple to owl vector type (w/ size sanity check) */
  inline vec3f make_vec3f(const std::vector<float> &vec)
  {
    if (vec.size() != 3)
      throw std::runtime_error("the tuple passed didn't have the"
                               " expected number of elements (3)");
    return vec3f(vec[0],vec[1],vec[2]);
  }
  
}
