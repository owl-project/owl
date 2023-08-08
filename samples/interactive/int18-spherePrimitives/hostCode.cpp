// ======================================================================== //
// Copyright 2019-2022 Ingo Wald                                            //
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
#include "owlViewer/InspectMode.h"
#include "owlViewer/OWLViewer.h"
#include "owl/common/math/LinearSpace.h"
#include <random>

#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"

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

const vec3f camera_init_lookFrom(0.f,0.f,10.f);
const vec3f camera_init_lookAt(0.f,0.f,0.f);
const vec3f camera_init_lookUp(0.f,1.f,3.f);
const float camera_init_cosFovy = 0.66f;

struct Viewer : public owl::viewer::OWLViewer
{
  Viewer();

  /*! creates a simple curves geometry similar to the 'optixCurves'
      example in the OptiX SDK */
  void createScene();
  
  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;

      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  OWLRayGen  rayGen  { 0 };
  OWLParams  lp      { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
  OWLGroup   spheresGeomGroup;
  OWLBuffer  accumBuffer { 0 };
  int accumID { 0 };

  // the spheres model we're going to render
  std::vector<float> radii;
  std::vector<vec3f> vertices;
  std::vector<vec3f> colors;
  OWLBuffer radiiBuffer, verticesBuffer, colorsBuffer;
  
  bool sbtDirty = true;
};

/*! window notifies us that we got resized */
void Viewer::resize(const vec2i &newSize)
{
  if (!accumBuffer)
    accumBuffer = owlDeviceBufferCreate(context,OWL_FLOAT4,1,nullptr);
  owlBufferResize(accumBuffer,newSize.x*newSize.y);
  owlParamsSetBuffer(lp,"accumBuffer",accumBuffer);
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
  accumID = 0;

  float focal_distance = length(lookAt-lookFrom);
  focal_distance = fmaxf(focal_distance, 1e-2f);
  float focal_scale = 10.f;

  // ----------- set variables  ----------------------------
  owlRayGenSet1ul   (rayGen,"fbPtr",        (uint64_t)fbPointer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  // DoF camera setup
  owlRayGenSet1f    (rayGen,"camera.aperture_radius",.15f);
  owlRayGenSet1f    (rayGen,"camera.focal_scale",focal_scale);

  sbtDirty = true;
}

static float randomFloat01()
{
    return static_cast <float> (std::rand()) / static_cast <float> (RAND_MAX);
}

void Viewer::createScene()
{
  for (int i = 0; i < 500; i++)
  {
      float radius = 0.05f + (randomFloat01() * 0.2f);
      radii.push_back( radius );

      float size = 6.0f;
      float x = (randomFloat01() * size) - (size * 0.5f);
      float y = (randomFloat01() * size) - (size * 0.5f);
      float z = (randomFloat01() * size) - (size * 0.5f);
      vertices.push_back( vec3f( x, y, z ) );

      float red = randomFloat01();
      colors.push_back( vec3f( red, 1.0f, 1.0f - red) );
  }
}

Viewer::Viewer()
{
  setTitle("sample viewer: int18-spherePrimitives");
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  owlContextSetRayTypeCount(context,2);
  owlEnableSpheres(context);
  
  createScene();
  
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry types
  // -------------------------------------------------------
  OWLVarDecl spheresGeomVars[] = {
    { "tint", OWL_BUFPTR, OWL_OFFSETOF(SpheresGeomData,tint)},
    //
    { "material.Ka", OWL_FLOAT3, OWL_OFFSETOF(SpheresGeomData,material.Ka) },
    { "material.Kd", OWL_FLOAT3, OWL_OFFSETOF(SpheresGeomData,material.Kd) },
    { "material.Ks", OWL_FLOAT3, OWL_OFFSETOF(SpheresGeomData,material.Ks) },
    { "material.reflectivity", OWL_FLOAT3, OWL_OFFSETOF(SpheresGeomData,material.reflectivity) },
    { "material.phong_exp", OWL_FLOAT, OWL_OFFSETOF(SpheresGeomData,material.phong_exp) },
    { nullptr }
  };
  OWLGeomType spheresGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_SPHERES,
                        sizeof(SpheresGeomData),
                        spheresGeomVars,-1);
  owlGeomTypeSetClosestHit(spheresGeomType,RADIANCE_RAY_TYPE,
                           module,"SpheresGeom");
  owlGeomTypeSetAnyHit(spheresGeomType,SHADOW_RAY_TYPE,
                       module,"SpheresGeom");

  // Call this so we have the bounds progs available
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  verticesBuffer 
    = owlDeviceBufferCreate(context,OWL_FLOAT3,vertices.size(),vertices.data());
  radiiBuffer 
    = owlDeviceBufferCreate(context,OWL_FLOAT,radii.size(),radii.data());
  colorsBuffer 
    = owlDeviceBufferCreate(context,OWL_FLOAT3,colors.size(),colors.data());
  
  OWLGeom spheresGeom = owlGeomCreate(context,spheresGeomType);
  owlSpheresSetVertices(spheresGeom,vertices.size(),verticesBuffer,radiiBuffer);

#if 1
  owlGeomSetBuffer(spheresGeom, "tint", colorsBuffer);
  owlGeomSet3f(spheresGeom,"material.Ka",.15f,.15f,.15f);
  owlGeomSet3f(spheresGeom,"material.Kd",.25f,.25f,.25f);
  owlGeomSet3f(spheresGeom,"material.Ks",.4f,.4f,.4f);
  owlGeomSet3f(spheresGeom,"material.reflectivity",0.f,0.f,0.f);
  owlGeomSet1f(spheresGeom,"material.phong_exp",20.f);
#else
  owlGeomSet3f(spheresGeom,"material.Ka",.35f,.35f,.35f);
  owlGeomSet3f(spheresGeom,"material.Kd",.5f,.5f,.5f);
  owlGeomSet3f(spheresGeom,"material.Ks",1.f,1.f,1.f);
  owlGeomSet3f(spheresGeom,"material.reflectivity",0.f,0.f,0.f);
  owlGeomSet1f(spheresGeom,"material.phong_exp",1.f);
#endif
  spheresGeomGroup = owlSphereGeomGroupCreate(context,1,&spheresGeom);
  owlGroupBuildAccel(spheresGeomGroup);

  world
    = owlInstanceGroupCreate(context,1,
                             &spheresGeomGroup,
                             nullptr,
                             nullptr,
                             OWL_MATRIX_FORMAT_OWL);
  owlGroupBuildAccel(world);

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss prog
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "bg_color", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,bg_color)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg
    = owlMissProgCreate(context,module,"miss",sizeof(MissProgData),
                        missProgVars,-1);

  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"bg_color",owl3f{.34f,.55f,.85f});

  // -------------------------------------------------------
  // set up launch params
  // -------------------------------------------------------
  OWLVarDecl launchParamsVars[] = {
    { "world",         OWL_GROUP,  OWL_OFFSETOF(LaunchParams,world)},
    { "accumBuffer",   OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,accumBuffer) },
    { "accumID",   OWL_INT, OWL_OFFSETOF(LaunchParams,accumID) },
    { "numLights",     OWL_INT, OWL_OFFSETOF(LaunchParams,numLights)},
    { "lights",        OWL_BUFPTR, OWL_OFFSETOF(LaunchParams,lights)},
    { "ambient_light_color", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,ambient_light_color)},
    { "scene_epsilon", OWL_FLOAT, OWL_OFFSETOF(LaunchParams,scene_epsilon)},
    { nullptr/* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  lp = owlParamsCreate(context,sizeof(LaunchParams),launchParamsVars,-1);

  owlParamsSetGroup (lp,"world",        world);

  /* light sources */
  owlParamsSet1i(lp,"numLights",2);
  BasicLight lights[] = {
    { vec3f( -30.0f, -10.0f, 80.0f ), vec3f( 1.0f, 1.0f, 1.0f ) },
    { vec3f(  10.0f,  30.0f, 20.0f ), vec3f( 1.0f, 1.0f, 1.0f ) }
  };
  OWLBuffer lightBuffer
    = owlDeviceBufferCreate(context,OWL_USER_TYPE(BasicLight),2,&lights);
  owlParamsSetBuffer(lp,"lights",lightBuffer);
  owlParamsSet3f(lp,"ambient_light_color",.4f,.4f,.4f);
  owlParamsSet1f(lp,"scene_epsilon",1e-2f);

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { "camera.aperture_radius",    OWL_FLOAT, OWL_OFFSETOF(RayGenData,camera.aperture_radius)},
    { "camera.focal_scale",    OWL_FLOAT, OWL_OFFSETOF(RayGenData,camera.focal_scale)},
    { nullptr/* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

  /* camera and frame buffer get set in resiez() and cameraChanged() */
  owlRayGenSetGroup (rayGen,"world",        world);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  owlBuildPrograms(context);
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
  owlParamsSet1i(lp,"accumID",accumID);
  accumID++;
  owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
}


int main(int ac, char **av)
{
#if !OWL_CAN_DO_SPHERES
  std::cerr << OWL_TERMINAL_RED
            << "You tried to run a sample that requires support for 'sphere' primitives;\n"
            << "but OWL supports spheres only when compiled with OptiX >= 7.5.\n"
            << "please re-build with a newer version of OptiX, and re-run\n";
  exit(0);
#endif
  std::string arg1;
  if (ac>1) {
    arg1 = std::string(av[1]);
    if (arg1=="-h") {
      std::cout << "Usage: " << av[0] << "[args]\n";
      std::cout << "w/ args:" << std::endl;
      std::cout << "  -r <r0> <r1>    : begin and end radius\n";
      exit(EXIT_SUCCESS);
    }
  }

  Viewer viewer;
  viewer.camera.setOrientation(camera_init_lookFrom,
                               camera_init_lookAt,
                               camera_init_lookUp,
                               owl::viewer::toDegrees(acosf(camera_init_cosFovy)));
  viewer.enableFlyMode();
  viewer.enableInspectMode(viewer::OWLViewer::Arcball,
                           owl::box3f(vec3f(-10.f),vec3f(+10.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
