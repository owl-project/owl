// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include "qbvh.h"
#include "tiny_obj_loader.h"

namespace gdt {
  namespace qbvh {

    typedef vec3i PrimT;
    
    template<int NUM_CHILDREN>
    void testBVH(const size_t numTriangles, const vec3i *index, const vec3f *vertex)
    {
      BVH<NUM_CHILDREN> bvh;
      build(bvh,numTriangles,
            [&](size_t primID){ 
              box3f bounds;
              // PRINT(index[primID]);
              bounds.extend(vertex[index[primID].x]);
              bounds.extend(vertex[index[primID].y]);
              bounds.extend(vertex[index[primID].z]);
              return bounds;
            });
      std::cout << "done building " << NUM_CHILDREN << "-wide BVH over " << numTriangles << " triangles" << std::endl;
    }
    
    extern "C" int main(int ac,char**av) {
      if (ac != 2) exit(1);

      std::vector<tinyobj::shape_t> shapes;
      std::vector<tinyobj::material_t> materials;
      std::string err = "";
      tinyobj::LoadObj(shapes,materials,err,av[1],"");
      
      PRINT(shapes.size());
      PRINT(materials.size());
      PRINT(shapes[0].mesh.positions.size());
      PRINT(shapes[0].mesh.indices.size());

      size_t numTriangles = shapes[0].mesh.indices.size()/3;
      vec3i *index = (vec3i*)&shapes[0].mesh.indices[0];
      vec3f *vertex = (vec3f*)&shapes[0].mesh.positions[0];

      PRINT(numTriangles);
      testBVH<2>(numTriangles,index,vertex);
      testBVH<4>(numTriangles,index,vertex);
      testBVH<6>(numTriangles,index,vertex);
      testBVH<8>(numTriangles,index,vertex);
      testBVH<12>(numTriangles,index,vertex);
      testBVH<16>(numTriangles,index,vertex);
    }
  }
}
