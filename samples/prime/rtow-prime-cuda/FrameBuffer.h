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

#include "Materials.h"

namespace samples {
  
  struct FrameBuffer {
    FrameBuffer(const vec2i &size)
      : size(size), pixels(size.x*size.y)
    {}
    
    const vec2i        size;
    std::vector<vec3f> pixels;
  };

  void savePNG(const std::string &fileName,
               FrameBuffer &image);
  
}
