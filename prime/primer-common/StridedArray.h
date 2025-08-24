// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

namespace primer {
  
  /*! helper class to allow c-style array access to strided arrays
      (primarily used for on-the-fly SoA<->AoS conversions */
  template<typename T>
  struct StridedArray {
    StridedArray(T *base, size_t stride) : base(base), stride(stride) {}
    inline operator bool() const { return base != nullptr; }
    
    inline T &operator[](size_t idx)
    {
      uint8_t *ptr = (uint8_t*)base;
      ptr += stride*idx;
      return *(T*)ptr;
    }
    
    T *base;
    size_t stride;
  };
}

  
