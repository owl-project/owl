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

#define BOUNDS_ON_HOST 0

#include "ll/DeviceGroup.h"
#include "deviceCode.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << GDT_TERMINAL_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << GDT_TERMINAL_DEFAULT;

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

  owl::ll::DeviceGroup::SP ll
    = owl::ll::DeviceGroup::create();

  LOG("building pipeline ...");
  std::cout << GDT_TERMINAL_DEFAULT;

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  ll->allocModules(1);
  ll->setModule(0,ptxCode);
  ll->buildModules();
  
  enum { SPHERE_GEOM_TYPE=0,NUM_GEOM_TYPES };
  ll->allocGeomTypes(NUM_GEOM_TYPES);
  ll->geomTypeCreate(SPHERE_GEOM_TYPE,sizeof(SphereGeomData));
  ll->setGeomTypeClosestHit(/*geom type ID*/SPHERE_GEOM_TYPE,
                            /*ray type  */0,
                            /*module:*/0,
                            "Sphere");
  ll->setGeomTypeIntersect(/*geom type ID*/SPHERE_GEOM_TYPE,
                           /*ray type  */0,
                           /*module:*/0,
                           "Sphere");

  ll->setGeomTypeBoundsProgDevice(/*program ID*/0,
                                  /*module:*/0,
                                  "Sphere",
                                  sizeof(SphereGeomData));
  ll->allocRayGens(1);
  ll->setRayGen(/*program ID*/0,
                /*module:*/0,
                "simpleRayGen",
                sizeof(RayGenData));

  ll->allocMissProgs(1);
  ll->setMissProg(/*program ID*/0,
                  /*module:*/0,
                  "miss",
                  sizeof(MissProgData));
  ll->buildPrograms();
  ll->createPipeline();

  LOG("building geometries ...");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { FRAME_BUFFER=0,
         NUM_BUFFERS };
  ll->reallocBuffers(NUM_BUFFERS);
  ll->createHostPinnedBuffer(FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));

  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  ll->reallocGeoms(8);
  for (int i=0;i<8;i++) {
    ll->createUserGeom(/* geom ID    */i,
                       /* type/PG ID */0,
                       /* numprims   */1);
  }

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
  
  enum { SPHERES_GROUP=0,NUM_GROUPS };
  ll->reallocGroups(NUM_GROUPS);
  int geomsInGroup[] = { 0,1,2,3,4,5,6,7 };
  ll->createUserGeomGroup(/* group ID */SPHERES_GROUP,
                          /* geoms in group, pointer */ geomsInGroup,
                          /* geoms in group, count   */ 8);
  ll->groupBuildPrimitiveBounds
    (SPHERES_GROUP,sizeof(SphereGeomData),
     [&](uint8_t *output, int devID, int geomID, int childID) {
      SphereGeomData &self = *(SphereGeomData*)output;
      self.center = sphereCenters[geomID];
      self.radius = sphereRadius;
    });
  ll->groupBuildAccel(SPHERES_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  ll->sbtHitProgsBuild
    ([&](uint8_t *output,int devID,int geomID,int childID) {
      SphereGeomData &self = *(SphereGeomData*)output;
      self.center = sphereCenters[geomID];
      self.radius = sphereRadius;
      self.color  = gdt::randomColor(geomID);
    });
  
  // ----------- build miss prog(s) -----------
  ll->sbtMissProgsBuild
    ([&](uint8_t *output,
         int devID,
         int rayType) {
      ((MissProgData*)output)->color0 = vec3f(.8f,0.f,0.f);
      ((MissProgData*)output)->color1 = vec3f(.8f,.8f,.8f);
    });
  
  // ----------- build raygens -----------
  ll->sbtRayGensBuild
    ([&](uint8_t *output,
         int devID,
         int rgID) {
      RayGenData *rg = (RayGenData*)output;
      rg->deviceIndex   = devID;
      rg->deviceCount = ll->getDeviceCount();
      rg->fbSize = fbSize;
      rg->fbPtr  = (uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,devID);
      rg->world  = ll->groupGetTraversable(SPHERES_GROUP,devID);

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
  ll->launch(0,fbSize);
  // todo: explicit sync?
  
  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb = (const uint32_t*)ll->bufferGetPointer(FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying devicegroup ...");
  owl::ll::DeviceGroup::destroy(ll);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
