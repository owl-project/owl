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

#include "Ray.h"

namespace primer {

  struct Model;
  struct Geom;
  struct Group;
  
  struct Context {
    virtual ~Context() {}
    
    /*! create a mesh from vertex array and index array */
    virtual Geom *createTriangles(uint64_t userID,
                                  /* vertex array */
                                  const vec3f *vertices,
                                  size_t numVertices,
                                  size_t sizeOfVertexInBytes,
                                  /* index array */
                                  const vec3i *indices,
                                  size_t numIndices,
                                  size_t sizeOfIndexStructInBytes) = 0;

    virtual Group *createGroup(std::vector<OPGeom> &geoms) = 0;
  

    virtual Model *createModel(const std::vector<OPGroup>  &groups,
                               const std::vector<affine3f> &xfms) = 0;
    
    static Context *createOffloadContext(int gpuID);
    static Context *createEmbreeContext();

    /*! enables logging/debugging messages if true, suppresses them if
      false */
    static bool logging;
  };
  
#define LOG(a) if (Context::logging) std::cout << OWL_TERMINAL_LIGHT_BLUE << a << OWL_TERMINAL_DEFAULT << std::endl;

}
