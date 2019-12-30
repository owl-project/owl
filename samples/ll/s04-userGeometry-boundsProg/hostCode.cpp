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
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "ll04-userGeometry-boundsProg.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

const vec3f sphereCenters[8] = {
  { -1.f, -1.f, -1.f },
  { -1.f, -1.f, +1.f },
  { -1.f, +1.f, -1.f },
  { -1.f, +1.f, +1.f },
  { +1.f, -1.f, -1.f },
  { +1.f, -1.f, +1.f },
  { +1.f, +1.f, -1.f },
  { +1.f, +1.f, +1.f },
};
const float sphereRadius = 0.6f;

int main(int ac, char **av)
{
  LOG("ll example '" << av[0] << "' starting up");

  LLOContext llo = lloContextCreate(nullptr,0);

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  LOG("building pipeline ...");
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);
  
  enum { SPHERE_GEOM_TYPE=0,NUM_GEOM_TYPES };
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  lloGeomTypeCreate(llo,SPHERE_GEOM_TYPE,sizeof(SphereGeomData));
  lloGeomTypeClosestHit(llo,/*geom type ID*/SPHERE_GEOM_TYPE,
                        /*ray type  */0,
                        /*module:*/0,
                        "Sphere");
  lloGeomTypeIntersect(llo,/*geom type ID*/SPHERE_GEOM_TYPE,
                       /*ray type  */0,
                       /*module:*/0,
                       "Sphere");

  lloGeomTypeBoundsProgDevice(llo,/*program ID*/0,
                                  /*module:*/0,
                                  "Sphere",
                                  sizeof(SphereGeomData));
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

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { FRAME_BUFFER=0,
         NUM_BUFFERS };
  lloAllocBuffers(llo,NUM_BUFFERS);
  lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y*sizeof(uint32_t));

  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  lloAllocGeoms(llo,8);
  for (int i=0;i<8;i++) {
    lloUserGeomCreate(llo,/* geom ID    */i,
                       /* type/PG ID */0,
                       /* numprims   */1);
  }

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
  
  enum { SPHERES_GROUP=0,NUM_GROUPS };
  lloAllocGroups(llo,NUM_GROUPS);
  int geomsInGroup[] = { 0,1,2,3,4,5,6,7 };
  lloUserGeomGroupCreate(llo,
                         /* group ID */SPHERES_GROUP,
                         /* geoms in group, pointer */ geomsInGroup,
                         /* geoms in group, count   */ 8);
  lloGroupBuildPrimitiveBounds
    (llo,
     SPHERES_GROUP,sizeof(SphereGeomData),
     [&](uint8_t *output, int devID, int geomID, int childID) {
      SphereGeomData &self = *(SphereGeomData*)output;
      self.center = sphereCenters[geomID];
      self.radius = sphereRadius;
    });
  lloGroupAccelBuild(llo,SPHERES_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  lloSbtHitProgsBuild
    (llo,
     [&](uint8_t *output,int devID,int geomID,int childID) {
      SphereGeomData &self = *(SphereGeomData*)output;
      self.center = sphereCenters[geomID];
      self.radius = sphereRadius;
      self.color  = owl::randomColor(geomID);
    });
  
  // ----------- build miss prog(s) -----------
  lloSbtMissProgsBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rayType) {
      ((MissProgData*)output)->color0 = vec3f(.8f,0.f,0.f);
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
      rg->world  = lloGroupGetTraversable(llo,SPHERES_GROUP,devID);

      // compute camera frame:
      vec3f &pos = rg->camera.pos;
      vec3f &d00 = rg->camera.dir_00;
      vec3f &ddu = rg->camera.dir_du;
      vec3f &ddv = rg->camera.dir_dv;
      float aspect = fbSize.x / float(fbSize.y);
      pos = lookFrom;
      d00 = normalize(lookAt-lookFrom);
      ddu = cosFovy * aspect * normalize(cross(d00,lookUp));
      ddv = cosFovy * normalize(cross(ddu,d00));
      d00 -= 0.5f * ddu;
      d00 -= 0.5f * ddv;
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
