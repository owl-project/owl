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

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                            \
  std::cout << GDT_TERMINAL_BLUE;                               \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << GDT_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const int NUM_VERTICES = 5;
vec3f vertices[NUM_VERTICES] =
  {
    { -0.5f,-0.5f,-0.5f },
    { +0.5f,-0.5f,-0.5f },
    { +0.5f,+0.5f,-0.5f },
    { -0.5f,+0.5f,-0.5f },
    { 0.0f,0.0f,+0.5f },
  };

const int NUM_INDICES = 6;
vec3i indices[NUM_INDICES] =
  {
    { 0,1,3 }, { 1,2,3 },
    { 0,4,1 }, { 0,3,4 },
    { 3,2,4 }, { 1,4,2 },
  };

const char *outFileName = "ll08-sierpinski.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(2.f,1.3f,.8f);
const vec3f lookAt(0.f,0.f,-.2f);
const vec3f lookUp(0.f,0.f,1.f);
const float fovy = 30.f;

int main(int ac, char **av)
{
  uint32_t numLevels = 8;
  LOG("ll example '" << av[0] << "' starting up");

  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg == "--num-levels" || arg == "-nl")
      numLevels = std::atoi(av[++i]);
    else
      throw std::runtime_error("unknown cmdline argument '"+arg+"'");
  }
  if (numLevels < 1)
    throw std::runtime_error("num levels must be 1 or greater");
  
  LLOContext llo = lloContextCreate(nullptr,0);
  lloSetMaxInstancingDepth(llo,numLevels);
  

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  LOG("building pipeline ...");
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);
  
  enum { PYRAMID_GEOM_TYPE=0,NUM_GEOM_TYPES };
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  lloGeomTypeCreate(llo,PYRAMID_GEOM_TYPE,sizeof(LambertianPyramidMesh));
  lloGeomTypeClosestHit(llo,/*program ID*/PYRAMID_GEOM_TYPE,
                            /*ray type  */0,
                            /*module:*/0,
                            "PyramidMesh");
  
  lloAllocRayGens(llo,1);
  lloRayGenCreate(llo,/*program ID*/0,
                /*module:*/0,
                "simpleRayGen",
                sizeof(RayGenData));
  
  lloAllocMissProgs(llo,1);
  lloMissProgCreate(llo,/*program ID*/0,
                  /*module:*/0,
                  "miss",
                  sizeof(MissProgData));
  lloBuildPrograms(llo);
  lloCreatePipeline(llo);

  LOG("building geometries ...");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################
  auto lambertian = Lambertian();
  std::vector<Lambertian> lambertianPyramids;
  Lambertian green;
  green.albedo = owl::vec3f(0,.7f,0);
  lambertianPyramids.push_back(green);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { LAMBERTIAN_PYRAMIDS_MATERIAL_BUFFER=0,
         VERTEX_BUFFER,
         INDEX_BUFFER,
         FRAME_BUFFER,
         NUM_BUFFERS };
  lloAllocBuffers(llo,NUM_BUFFERS);
  lloDeviceBufferCreate(llo,LAMBERTIAN_PYRAMIDS_MATERIAL_BUFFER,
                        lambertianPyramids.size()
                        *sizeof(lambertianPyramids[0]),
                        lambertianPyramids.data());
  lloDeviceBufferCreate(llo,VERTEX_BUFFER,NUM_VERTICES*sizeof(vec3f),vertices);
  lloDeviceBufferCreate(llo,INDEX_BUFFER,NUM_INDICES*sizeof(vec3i),indices);
  lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y*sizeof(uint32_t));
  
  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  enum { PYRAMID_GEOM=0,NUM_GEOMS };
  lloAllocGeoms(llo,NUM_GEOMS);
  lloTrianglesGeomCreate(llo,/* geom ID    */PYRAMID_GEOM,
                          /* type/PG ID */PYRAMID_GEOM_TYPE);
  lloTrianglesGeomSetVertexBuffer(llo,/* geom ID   */PYRAMID_GEOM,
                                   /* buffer ID */VERTEX_BUFFER,
                                   /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  lloTrianglesGeomSetIndexBuffer(llo,/* geom ID   */PYRAMID_GEOM,
                                  /* buffer ID */INDEX_BUFFER,
                                  /* meta info */NUM_INDICES,sizeof(vec3i),0);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
  int WORLD_GROUP = numLevels - 1;
  
  // enum { TRIANGLES_GROUP=0,PYRAMID_GROUP_LVL_1,WORLD_GROUP,NUM_GROUPS };
  lloAllocGroups(llo,numLevels);
  int geomsInGroup[] = { 0 };
  lloTrianglesGeomGroupCreate(llo,/* group ID */0,
                               /* geoms in group, pointer */ geomsInGroup,
                               /* geoms in group, count   */ 1); 
  lloGroupAccelBuild(llo,0);

  auto make_sierpinski = [&](int parent_level, int child_level){
    int groupsInWorldGroup[]
    = { child_level,
        child_level,
        child_level,
        child_level,
        child_level, };
    lloInstanceGroupCreate(llo,
                           /* group ID */parent_level,
                           /* geoms in group, pointer */ groupsInWorldGroup,
                           /* geoms in group, count   */ 5);
    auto a
    = owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
    * owl::affine3f::translate(owl::vec3f(-.5f, -.5f, -.5f));
    auto b
    = owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
    * owl::affine3f::translate(owl::vec3f(+.5f, -.5f, -.5f));
    auto c
    = owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
    * owl::affine3f::translate(owl::vec3f(-.5f, +.5f, -.5f));
    auto d
    = owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))
    * owl::affine3f::translate(owl::vec3f(+.5f, +.5f, -.5f));
    auto e
    = owl::affine3f::scale(owl::vec3f(.5f,.5f,.5f))    
    * owl::affine3f::translate(owl::vec3f(0.0f, 0.0, +.5f));
    
    lloInstanceGroupSetTransform(llo,parent_level,0,(const float *)&a);
    lloInstanceGroupSetTransform(llo,parent_level,1,(const float *)&b);
    lloInstanceGroupSetTransform(llo,parent_level,2,(const float *)&c);
    lloInstanceGroupSetTransform(llo,parent_level,3,(const float *)&d);
    lloInstanceGroupSetTransform(llo,parent_level,4,(const float *)&e);
    lloGroupAccelBuild(llo,parent_level);
  };

  for (uint32_t i = 1; i < numLevels; ++i) {
    make_sierpinski(i,i-1);
  }  

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  lloSbtHitProgsBuild
    (llo,
     [&](uint8_t *output,int devID,int geomID,int rayID) {
      LambertianPyramidMesh &self = *(LambertianPyramidMesh*)output;
      self.material
           = (Lambertian *)lloBufferGetPointer(llo,LAMBERTIAN_PYRAMIDS_MATERIAL_BUFFER,devID);
      self.index  = (vec3i*)lloBufferGetPointer(llo,INDEX_BUFFER,devID);
      self.vertex = (vec3f*)lloBufferGetPointer(llo,VERTEX_BUFFER,devID);
    });
  
  // ----------- build miss prog(s) -----------
  lloSbtMissProgsBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rayType) {
      /* we don't have any ... */
      ((MissProgData*)output)->color0 = vec3f(.8f,.8f,.8f);
      ((MissProgData*)output)->color1 = vec3f(.8f,.8f,.8f);
    });
  
  // ----------- build raygens -----------
  lloSbtRayGensBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rgID) {
      RayGenData *rg = (RayGenData*)output;
      rg->deviceIndex   = devID;
      rg->deviceCount = lloGetDeviceCount(llo);
      rg->fbSize = fbSize;
      rg->fbPtr  = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
      rg->world  = lloGroupGetTraversable(llo,WORLD_GROUP,devID);

      // compute camera frame:
      const float vfov = fovy;
      const vec3f vup = lookUp;
      const float aspect = fbSize.x / float(fbSize.y);
      const float theta = vfov * ((float)M_PI) / 180.0f;
      const float half_height = tanf(theta / 2.0f);
      const float half_width = aspect * half_height;
      const float aperture = 0.f;
      const float focusDist = 10.f;
      const vec3f origin = lookFrom;
      const vec3f w = normalize(lookFrom - lookAt);
      const vec3f u = normalize(cross(vup, w));
      const vec3f v = cross(w, u);
      const vec3f lower_left_corner
        = origin - half_width * focusDist*u - half_height * focusDist*v - focusDist * w;
      const vec3f horizontal = 2.0f*half_width*focusDist*u;
      const vec3f vertical = 2.0f*half_height*focusDist*v;

      rg->camera.origin = origin;
      rg->camera.lower_left_corner = lower_left_corner;
      rg->camera.horizontal = horizontal;
      rg->camera.vertical = vertical;
    });
  LOG_OK("everything set up ...");

  // ##################################################################
  // now that everything is readly: launch it ....
  // ##################################################################
  
  LOG("trying to launch ...");
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);
  // todo: explicit sync?
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  lloContextDestroy(llo);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
