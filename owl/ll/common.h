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

/*! \file optix/common.h Creates a common set of includes, #defines,
  and helpers that will be visible across _all_ files, both host _and_
  device */

#pragma once

// device-safe parts of gdt
#include "gdt/math/vec.h"
#include "gdt/math/box.h"
#include "gdt/math/AffineSpace.h"

#include <string.h>
#include <set>
#include <map>
#include <vector>
#include <stack>
#include <typeinfo>
#include <mutex>
#include <atomic>
#include <sstream>

namespace owl {
  using gdt::vec2f;
  using gdt::vec3f;
  using gdt::vec2i;
  using gdt::vec3i;
  using gdt::box3f;
  using gdt::linear3f;
  using gdt::affine3f;

  template<size_t alignment>
  inline size_t smallestMultipleOf(size_t unalignedSize)
  {
    const size_t numBlocks = (unalignedSize+alignment-1)/alignment;
    return numBlocks*alignment;
  }
  
  inline void *addPointerOffset(void *ptr, size_t offset)
  {
    if (ptr == nullptr) return nullptr;
    return (void*)((unsigned char *)ptr + offset);
  }
    
}

#define IGNORING_THIS() std::cout << "## ignoring " << __PRETTY_FUNCTION__ << std::endl;
  
#define OWL_NOTIMPLEMENTED std::cerr << (std::string(__PRETTY_FUNCTION__)+" : not implemented") << std::endl; exit(1);

