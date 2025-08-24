// ======================================================================== //
// Copyright 2018-2025 Ingo Wald                                            //
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

// helper stuff to more easily create test geometry
#include "owl/common/math/AffineSpace.h"
// primer itself
#include "owl/owl_prime.h"
// std stuff
#include <vector>
#include <iostream>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <random>

using namespace owl::common;

unsigned char to255(float f)
{
  if (f <= 0.f) return 0;
  if (f >= 1.f) return 255;
  return (unsigned char)(f*255.9f);
}

void savePNG(const std::string &fileName,
             vec2i fbSize,
             const vec3f *pixels)
{
  std::vector<unsigned char> rgba;
  for (int iy=fbSize.y-1;iy>=0;--iy) 
    for (int ix=0;ix<fbSize.x;ix++) {
      vec3f pixel = pixels[ix+fbSize.x*iy];
      rgba.push_back(to255(pixel.x));
      rgba.push_back(to255(pixel.y));
      rgba.push_back(to255(pixel.z));
      rgba.push_back(255);
    }
  std::cout << "#owl-prime.rtWeekend: writing image " << fileName << std::endl;
  stbi_write_png(fileName.c_str(),fbSize.x,fbSize.y,4,
                 rgba.data(),fbSize.x*sizeof(uint32_t));
}

inline float randomFloat()
{
  static std::random_device rd;  // Will be used to obtain a seed for the random number engine
  static std::mt19937 gen(rd()); // Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f randomPoint()
{
  return vec3f(randomFloat(),randomFloat(),randomFloat());
}

inline vec3f randomDirection()
{
  while (true) {
    vec3f v = 2.f*randomPoint()-vec3f(1.f);
    if (dot(v,v) <= 1.f)
      return normalize(v);
  }
}

int main(int, char **)
{
  std::cout << "creating primer context" << std::endl;
  OPContext context
    = opContextCreate(OP_CONTEXT_DEFAULT,0);

  // a simple box:
  vec3f vertices[]
    = {
       vec3f(0.f,0.f,0.f),
       vec3f(1.f,0.f,0.f),
       vec3f(0.f,1.f,0.f),
       vec3f(1.f,1.f,0.f),
       vec3f(0.f,0.f,1.f),
       vec3f(1.f,0.f,1.f),
       vec3f(0.f,1.f,1.f),
       vec3f(1.f,1.f,1.f)
  };

  int indices[] = {0,1,3, 2,3,0,
                   5,7,6, 5,6,4,
                   0,4,5, 0,5,1,
                   2,3,7, 2,7,6,
                   1,5,7, 1,7,3,
                   4,0,2, 4,2,6
  };

  std::cout << "creating a simple box mesh" << std::endl;
  OPGeom mesh
    = opMeshCreate(context,
                   /* user-supplied geometry ID */ 0,
                   (float*)vertices,8,sizeof(vec3f),
                   indices,12,3*sizeof(int));

  std::cout << "creating a group that can be instantiated" << std::endl;
  OPGroup group
    = opGroupCreate(context,&mesh,1);

  std::vector<OPTransform> transforms;
  std::vector<OPGroup> groups;
  for (int i=0;i<100;i++) {
    affine3f transform;
    transform.p = 4.f*(randomPoint()-vec3f(.5f));
    transform.l = frame(randomDirection());
    transforms.push_back((const OPTransform &)transform);
    groups.push_back(group);
  }

  std::cout << "creating model..." << std::endl;
  OPModel model
    = opModelCreate(context,
                    groups.data(),
                    transforms.data(),
                    transforms.size());

  std::cout << "generating rays..." << std::endl;
  vec3f up(0,1,0);
  vec3f at(0,0,0);
  vec3f from(-3,-2,-1);

  vec2i fbSize(800,600);
  vec3f dir = normalize(at-from);
  float imagePlaneHeight = 10.f;
  vec3f horiz = (imagePlaneHeight * fbSize.x/float(fbSize.y)) * normalize(cross(dir,up));
  vec3f vert  = imagePlaneHeight * normalize(cross(horiz,dir));

  std::vector<OPRay> rays;
  for (int iy=0;iy<fbSize.y;iy++)
    for (int ix=0;ix<fbSize.x;ix++) {
      float du = (ix+.5f)/fbSize.x;
      float dv = (iy+.5f)/fbSize.y;
      OPRay ray;
      (vec3f&)ray.origin = from + (du-.5f) * horiz + (dv-.5f) * vert;
      (vec3f&)ray.direction = dir;
      ray.tMin = 0.f;
      ray.tMax = INFINITY;
      rays.push_back(ray);
    }
  std::cout << "tracing rays..." << std::endl;
  std::vector<OPHit> hits(rays.size());
  opTrace(model,rays.size(),rays.data(),hits.data(),OP_TRACE_FLAGS_DEFAULT);

  std::vector<vec3f> pixels(rays.size());
  // ------------------------------------------------------------------
  std::cout << "creating 'instID' image" << std::endl;
  for (int i=0;i<rays.size();i++)
    pixels[i] = randomColor(hits[i].instID);
  savePNG("opSampleInstances_instIDs.png",fbSize,pixels.data());
  // ------------------------------------------------------------------
  std::cout << "creating 'primID' image" << std::endl;
  for (int i=0;i<rays.size();i++)
    pixels[i] = randomColor(hits[i].primID);
  savePNG("opSampleInstances_primIDs.png",fbSize,pixels.data());
}
