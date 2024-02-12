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

// This program sets up a single geometric object, a mesh for a cube, and
// its acceleration structure, then ray traces it.

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "deviceCode.h"
// viewer base class, for window and user interaction
#include "owlViewer/OWLViewer.h"
#include "owl/common/math/AffineSpace.h"
#include <random>

using namespace owl::common;

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

struct Mesh {
  std::vector<vec3f> vertices;
  std::vector<vec2f> texCoords;
  std::vector<vec3i> indices;
};

// const vec2i fbSize(800,600);
const vec3f init_lookFrom(-4.f,+3.f,-2.f);
const vec3f init_lookAt(0.f,0.f,0.f);
const vec3f init_lookUp(0.f,1.f,0.f);
const float init_cosFovy = 0.66f;

const vec3i numBoxes(4);
const float worldSize = 1;

std::default_random_engine rndGen;
std::uniform_real_distribution<float> distribution_uniform(-1.0f,1.0f);
std::uniform_real_distribution<float> distribution_speed(.1f,.8f);
std::uniform_real_distribution<float> distribution_rot(-.1f,+.1f);
std::uniform_int_distribution<int> distribution_texSize(2,16);

inline vec3f getRandomDir()
{
  vec3f rotationAxis;
  do {
    rotationAxis.x = distribution_uniform(rndGen);
    rotationAxis.y = distribution_uniform(rndGen);
    rotationAxis.z = distribution_uniform(rndGen);
  } while (dot(rotationAxis,rotationAxis) > 1.f);
  return normalize(rotationAxis);
}

void getCenters(vec3f &pos0,
                vec3f &pos1,
                vec3i boxID)
{
  const vec3f rel = (vec3f(boxID)+.5f) / vec3f(numBoxes);
  const vec3f boxCenter = vec3f(-worldSize) + (2.f*worldSize)*rel;
  pos0 = boxCenter;

  const float speed  = distribution_speed(rndGen);
  const vec3f motion = speed * getRandomDir();
  pos1 = pos0+motion;
}

OWLGroup createBoxesGroup(OWLContext context,
                          OWLGeomType boundsGeomType,
                          const vec3i numBoxes)
{
  std::vector<vec3f> boxCenters;

  for (int iz=0;iz<numBoxes.z;iz++)
    for (int iy=0;iy<numBoxes.y;iy++)
      for (int ix=0;ix<numBoxes.x;ix++) {
        vec3f c0,c1;
        getCenters(c0,c1,vec3i(ix,iy,iz));
        boxCenters.push_back(c0);
        boxCenters.push_back(c1);
      }

  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,boxCenters.size(),boxCenters.data());  

  OWLGeom userGeom
    = owlGeomCreate(context,boundsGeomType);
  owlGeomSetPrimCount(userGeom,boxCenters.size() / 2);
  owlGeomSetBuffer(userGeom,"vertex",vertexBuffer);

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup userGeomGroup
    = owlUserGeomGroupCreate(context,1,&userGeom);

  owlGroupBuildAccel(userGeomGroup);

  return userGeomGroup;
}


struct Viewer : public owl::viewer::OWLViewer
{
  Viewer();

  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;

      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  bool sbtDirty = true;
  OWLRayGen  rayGen  { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
};

/*! window notifies us that we got resized */
void Viewer::resize(const vec2i &newSize)
{
  OWLViewer::resize(newSize);
  cameraChanged();
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
  const vec3f lookFrom = camera.getFrom();
  const vec3f lookAt = camera.getAt();
  const vec3f lookUp = camera.getUp();
  const float cosFovy = camera.getCosFovy();
  // ----------- compute variable values  ------------------
  vec3f camera_pos = lookFrom;
  vec3f camera_d00
    = normalize(lookAt-lookFrom);
  float aspect = fbSize.x / float(fbSize.y);
  vec3f camera_ddu
    = cosFovy * aspect * normalize(cross(camera_d00,lookUp));
  vec3f camera_ddv
    = cosFovy * normalize(cross(camera_ddu,camera_d00));
  camera_d00 -= 0.5f * camera_ddu;
  camera_d00 -= 0.5f * camera_ddv;

  // ----------- set variables  ----------------------------
  owlRayGenSet1ul   (rayGen,"fbPtr",        (uint64_t)fbPointer);
  // owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  vec3f lightDir = {1.f,1.f,1.f};
  owlRayGenSet3f    (rayGen,"lightDir",     (const owl3f&)lightDir);
  sbtDirty = true;
}

Viewer::Viewer()
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  owlEnableMotionBlur(context);
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl boundsGeomVars[] = {
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(BoundsGeomData,vertex)},
    { nullptr }
  };
  OWLGeomType boundsGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOM_USER,
                        sizeof(BoundsGeomData),
                        boundsGeomVars,-1);
  owlGeomTypeSetMotionBoundsProg(boundsGeomType, 
                                 module, "Bounds");
  owlGeomTypeSetIntersectProg(boundsGeomType, 0,
                              module, "Bounds");
  owlGeomTypeSetClosestHit(boundsGeomType, 0,
                           module, "BoundsMesh");

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
    // { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "lightDir",      OWL_FLOAT3, OWL_OFFSETOF(RayGenData,lightDir)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { nullptr/* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);
                      
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  OWLGroup group = createBoxesGroup(context,boundsGeomType,numBoxes);
  world = owlInstanceGroupCreate(context, 1, &group);
  owlGroupBuildAccel(world);

  /* camera and frame buffer get set in resiez() and cameraChanged() */
  owlRayGenSetGroup (rayGen,"world",        world);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  owlBuildPipeline(context);
  owlBuildSBT(context);
  sbtDirty = true;
}

void Viewer::render()
{
  if (sbtDirty) {
    owlBuildSBT(context);
    sbtDirty = false;
  }
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
}


int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  Viewer viewer;
  viewer.camera.setOrientation(init_lookFrom,
                               init_lookAt,
                               init_lookUp,
                               owl::viewer::toDegrees(acosf(init_cosFovy)));
  viewer.enableFlyMode();
  viewer.enableInspectMode(owl::box3f(vec3f(-1.f),vec3f(+1.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
