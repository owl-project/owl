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

#include <random>

#define LOG(message)                                            \
  std::cout << GDT_TERMINAL_BLUE;                               \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << GDT_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << GDT_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#ll.sample(main): " << message << std::endl;    \
  std::cout << GDT_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const char *outFileName = "ll05-rtow.png";
const vec2i fbSize(1600,800);
const vec3f lookFrom(13, 2, 3);
const vec3f lookAt(0, 0, 0);
const vec3f lookUp(0.f,1.f,0.f);
const float fovy = 20.f;

std::vector<DielectricSphere> dielectricSpheres;
std::vector<LambertianSphere> lambertianSpheres;
std::vector<MetalSphere>      metalSpheres;

inline float max3(float a, float b, float c)
{ return std::max(std::max(a,b),c); }

inline float rnd()
{
  static std::mt19937 gen(0); //Standard mersenne_twister_engine seeded with rd()
  static std::uniform_real_distribution<float> dis(0.f, 1.f);
  return dis(gen);
}

inline vec3f rnd3f() { return vec3f(rnd(),rnd(),rnd()); }

void createScene()
{
  lambertianSpheres.push_back({Sphere{vec3f(0.f, -1000.0f, -1.f), 1000.f},
                               Lambertian{vec3f(0.5f, 0.5f, 0.5f)}});
  
  for (int a = -11; a < 11; a++) {
    for (int b = -11; b < 11; b++) {
      float choose_mat = rnd();
      vec3f center(a + rnd(), 0.2f, b + rnd());
      if (choose_mat < 0.8f) 
        lambertianSpheres.push_back({Sphere{center, 0.2f},
                                     Lambertian{rnd3f()*rnd3f()}});
      else if (choose_mat < 0.95f) 
        metalSpheres.push_back({Sphere{center, 0.2f},
                                Metal{0.5f*(1.f+rnd3f()),0.5f*rnd()}});
      else 
        dielectricSpheres.push_back({Sphere{center, 0.2f},
                                     Dielectric{1.5f}});
    }
  }
  dielectricSpheres.push_back({Sphere{vec3f(0.f, 1.f, 0.f), 1.f},
                               Dielectric{1.5f}});
  lambertianSpheres.push_back({Sphere{vec3f(-4.f,1.f, 0.f), 1.f},
                               Lambertian{vec3f(0.4f, 0.2f, 0.1f)}});
  metalSpheres.push_back({Sphere{vec3f(4.f, 1.f, 0.f), 1.f},
                          Metal{vec3f(0.7f, 0.6f, 0.5f), 0.0f}});
}
  
int main(int ac, char **av)
{
  LOG("ll example '" << av[0] << "' starting up");

  LOG("creating the scene ...");
  createScene();
  LOG_OK("created scene:");
  LOG_OK(" num lambertian spheres: " << lambertianSpheres.size());
  LOG_OK(" num dielectric spheres: " << dielectricSpheres.size());
  LOG_OK(" num metal spheres     : " << metalSpheres.size());
  
  LLOContext llo = lloContextCreate(nullptr,0);

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################
  LOG("building pipeline ...");
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);
  
  enum { METAL_SPHERES_TYPE=0,
         DIELECTRIC_SPHERES_TYPE,
         LAMBERTIAN_SPHERES_TYPE,
         NUM_GEOM_TYPES };
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  lloGeomTypeCreate(llo,METAL_SPHERES_TYPE,sizeof(MetalSpheresGeom));
  lloGeomTypeCreate(llo,LAMBERTIAN_SPHERES_TYPE,sizeof(LambertianSpheresGeom));
  lloGeomTypeCreate(llo,DIELECTRIC_SPHERES_TYPE,sizeof(DielectricSpheresGeom));
  lloGeomTypeClosestHit(llo,/*geom type ID*/LAMBERTIAN_SPHERES_TYPE,
                            /*ray type  */0,
                            /*module:*/0,
                            "LambertianSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/LAMBERTIAN_SPHERES_TYPE,
                           /*ray type  */0,
                           /*module:*/0,
                           "LambertianSpheres");
  ll->setGeomTypeBoundsProgDevice(/*program ID*/LAMBERTIAN_SPHERES_TYPE,
                                  /*module:*/0,
                                  "LambertianSpheres",
                                  sizeof(LambertianSpheresGeom));

  lloGeomTypeClosestHit(llo,/*geom type ID*/DIELECTRIC_SPHERES_TYPE,
                            /*ray type  */0,
                            /*module:*/0,
                            "DielectricSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/DIELECTRIC_SPHERES_TYPE,
                           /*ray type  */0,
                           /*module:*/0,
                           "DielectricSpheres");
  ll->setGeomTypeBoundsProgDevice(/*program ID*/DIELECTRIC_SPHERES_TYPE,
                                  /*module:*/0,
                                  "DielectricSpheres",
                                  sizeof(DielectricSpheresGeom));
  
  lloGeomTypeClosestHit(llo,/*geom type ID*/METAL_SPHERES_TYPE,
                            /*ray type  */0,
                            /*module:*/0,
                            "MetalSpheres");
  lloGeomTypeIntersect(llo,/*geom type ID*/METAL_SPHERES_TYPE,
                           /*ray type  */0,
                           /*module:*/0,
                           "MetalSpheres");
  ll->setGeomTypeBoundsProgDevice(/*program ID*/METAL_SPHERES_TYPE,
                                  /*module:*/0,
                                  "MetalSpheres",
                                  sizeof(MetalSpheresGeom));

  ll->allocRayGens(1);
  ll->setRayGen(/*program ID*/0,
                /*module:*/0,
                "rayGen",
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
         LAMBERTIAN_SPHERES_BUFFER,
         DIELECTRIC_SPHERES_BUFFER,
         METAL_SPHERES_BUFFER,
         NUM_BUFFERS };
  ll->allocBuffers(NUM_BUFFERS);
  ll->hostPinnedBufferCreate(FRAME_BUFFER,fbSize.x*fbSize.y,sizeof(uint32_t));

  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  enum { LAMBERTIAN_SPHERES_GEOM=0,
         DIELECTRIC_SPHERES_GEOM,
         METAL_SPHERES_GEOM,
         NUM_GEOMS };
  ll->allocGeoms(NUM_GEOMS);
  ll->userGeomCreate(/* geom ID    */LAMBERTIAN_SPHERES_GEOM,
                     /* type/PG ID */LAMBERTIAN_SPHERES_TYPE,
                     /* numprims   */lambertianSpheres.size());
  ll->deviceBufferCreate(LAMBERTIAN_SPHERES_BUFFER,
                         lambertianSpheres.size(),
                         sizeof(lambertianSpheres[0]),
                         lambertianSpheres.data());
  ll->userGeomCreate(/* geom ID    */DIELECTRIC_SPHERES_GEOM,
                     /* type/PG ID */DIELECTRIC_SPHERES_TYPE,
                     /* numprims   */dielectricSpheres.size());
  ll->deviceBufferCreate(DIELECTRIC_SPHERES_BUFFER,
                         dielectricSpheres.size(),
                         sizeof(dielectricSpheres[0]),
                         dielectricSpheres.data());
  ll->userGeomCreate(/* geom ID    */METAL_SPHERES_GEOM,
                     /* type/PG ID */METAL_SPHERES_TYPE,
                     /* numprims   */metalSpheres.size());
  ll->deviceBufferCreate(METAL_SPHERES_BUFFER,
                         metalSpheres.size(),
                         sizeof(metalSpheres[0]),
                         metalSpheres.data());
  
  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################
  
  enum { WORLD_GROUP=0,
         NUM_GROUPS };
  ll->allocGroups(NUM_GROUPS);
  int geomsInGroup[] = {
                        LAMBERTIAN_SPHERES_GEOM,
                        DIELECTRIC_SPHERES_GEOM,
                        METAL_SPHERES_GEOM
  };
  ll->userGeomGroupCreate(/* group ID */WORLD_GROUP,
                          /* geoms in group, pointer */ geomsInGroup,
                          /* geoms in group, count   */ NUM_GEOMS);
  ll->groupBuildPrimitiveBounds
    (WORLD_GROUP,max3(sizeof(MetalSpheresGeom),
                      sizeof(DielectricSpheresGeom),
                      sizeof(LambertianSpheresGeom)),
     [&](uint8_t *output, int devID, int geomID, int childID) {
       switch(geomID) {
       case LAMBERTIAN_SPHERES_GEOM:
         ((LambertianSpheresGeom*)output)->prims
           = (LambertianSphere*)ll->bufferGetPointer(LAMBERTIAN_SPHERES_BUFFER,devID);
         break;
       case DIELECTRIC_SPHERES_GEOM:
         ((DielectricSpheresGeom*)output)->prims
           = (DielectricSphere*)ll->bufferGetPointer(DIELECTRIC_SPHERES_BUFFER,devID);
         break;
       case METAL_SPHERES_GEOM:
         ((MetalSpheresGeom*)output)->prims
           = (MetalSphere*)ll->bufferGetPointer(METAL_SPHERES_BUFFER,devID);
         break;
       default:
         assert(0);
       }
     });
  ll->groupBuildAccel(WORLD_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  ll->sbtHitProgsBuild
    ([&](uint8_t *output,int devID,int geomID,int childID) {
       switch(geomID) {
       case LAMBERTIAN_SPHERES_GEOM:
         ((LambertianSpheresGeom*)output)->prims
           = (LambertianSphere*)ll->bufferGetPointer(LAMBERTIAN_SPHERES_BUFFER,devID);
         break;
       case DIELECTRIC_SPHERES_GEOM:
         ((DielectricSpheresGeom*)output)->prims
           = (DielectricSphere*)ll->bufferGetPointer(DIELECTRIC_SPHERES_BUFFER,devID);
         break;
       case METAL_SPHERES_GEOM:
         ((MetalSpheresGeom*)output)->prims
           = (MetalSphere*)ll->bufferGetPointer(METAL_SPHERES_BUFFER,devID);
         break;
       default:
         assert(0);
       }
     });
  
  // ----------- build miss prog(s) -----------
  ll->sbtMissProgsBuild
    ([&](uint8_t *output,
         int devID,
         int rayType) {
       /* we don't have any ... */
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
       rg->world  = ll->groupGetTraversable(WORLD_GROUP,devID);

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
  lloContextDestroy(llo);
  
  LOG_OK("seems all went ok; app is done, this should be the last output ...");
}
