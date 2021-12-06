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

const vec3f camera_init_lookFrom(0.f,0.f,2.f);
const vec3f camera_init_lookAt(0.f,0.f,0.f);
const vec3f camera_init_lookUp(0.f,1.f,3.f);
const float camera_init_cosFovy = 0.66f;

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

  OWLRayGen  rayGen  { 0 };
  OWLParams  lp      { 0 };
  OWLContext context { 0 };
  OWLGroup   world   { 0 };
  OWLGroup   curvesGroups;
  OWLBuffer  accumBuffer { 0 };
  int accumID { 0 };

  // the curves model we're going to render
  std::vector<int>   segmentIndices;
  std::vector<float> widths;
  std::vector<vec3f> vertices;
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
  owlRayGenSet1f    (rayGen,"camera.aperture_radius",setup==SETUP_TRIANGLE ? .15f : .1f);
  owlRayGenSet1f    (rayGen,"camera.focal_scale",focal_scale);

  sbtDirty = true;
}

void Viewer::createScene()
{
  segmentIndices = std::vector<int>{ 0 };

            // Number of motion keys
  const int NUM_KEYS = 6;
  
  int    degree = 3;
  float  radius = 0.4f;
  for( int i = 0; i < NUM_KEYS; ++i ) {
    // move the y-coordinates based on cosine
    const float c = cosf(i / static_cast<float>(NUM_KEYS) * 2.0f * static_cast<float>(M_PI));
    switch( degree ) {
    case 1: {
      vertices.push_back( make_float3( -0.25f, -0.25f * c, 0.0f ) );
      widths.push_back( 0.3f );
      vertices.push_back( make_float3( 0.25f, 0.25f * c, 0.0f ) );
      widths.push_back( radius );
    } break;
    case 2: {
      vertices.push_back( make_float3( -1.5f, -2.0f * c, 0.0f ) );
      widths.push_back( .01f );
      vertices.push_back( make_float3( 0.0f, 1.0f * c, 0.0f ) );
      widths.push_back( radius );
      vertices.push_back( make_float3( 1.5f, -2.0f * c, 0.0f ) );
      widths.push_back( .01f );
    } break;
    case 3: {
      vertices.push_back( make_float3( -1.5f, -3.5f * c, 0.0f ) );
      widths.push_back( .01f );
      vertices.push_back( make_float3( -1.0f, 0.5f * c, 0.0f ) );
      widths.push_back( radius );
      vertices.push_back( make_float3( 1.0f, 0.5f * c, 0.0f ) );
      widths.push_back( radius );
      vertices.push_back( make_float3( 1.5f, -3.5f * c, 0.0f ) );
      widths.push_back( .01f );
    } break;
    default:
      SUTIL_ASSERT_MSG( false, "Curve degree must be in {1, 2, 3}." );
    }
  }
}

Viewer::Viewer()
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  owlContextSetRayTypeCount(context,2);

  createScene();
  
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry types
  // -------------------------------------------------------
  OWLVarDecl curvesGeomVars[] = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(CurvesGeomData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(CurvesGeomData,color1)},
    { nullptr }
  };
  OWLGeomType curvesGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_CURVES,
                        sizeof(PoolBallsGeomData),
                        poolBallsGeomVars,-1);
  owlGeomTypeSetClosestHit(poolBallsGeomType,RADIANCE_RAY_TYPE,
                           module,"CurvesGeom");

  // Call this so we have the bounds progs available
  owlBuildPrograms(context);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  verticesBuffer =
    = owlDeviceBufferCreate(context,OWL_FLOAT3,vertices.size(),vertices.data());
  widthsBuffer =
    = owlDeviceBufferCreate(context,OWL_FLOAT1,widths.size(),widths.data());
  segmentsBuffer =
    = owlDeviceBufferCreate(context,OWL_INT1,segments.size(),segments.data());

  OWLGeom curvesGeom = owlGeomCreate(context,curvesGeomType);
  owlCurvesSetControlPoints(curvesGeom,vertices.size(),verticesBuffer,widthsBuffers);
  owlCurvesSetSegments(curvesGeom,segments.size(),segmentsBuffer);

  groups[0] = owlUserGeomGroupCreate(context,1,&poolBallsGeom);
  owlGroupBuildAccel(groups[0]);


  vec3f anchor( -5.00f, -25.0f, 0.01f );
  vec3f v1( 20.0f,  0.0f, 0.01f );
  vec3f v2(  0.0f, 50.0f, 0.01f );

  vec3f normal = cross( v1, v2 );
  normal = normalize( normal );
  float d = dot( normal, anchor );
  v1 *= 1.0f/dot( v1, v1 );
  v2 *= 1.0f/dot( v2, v2 );
  vec4f plane( normal, d );

  vec2i res;
  int comp;
  std::string path(DATA_PATH);
  path += "/cloth.ppm";
  unsigned char* image = stbi_load(path.c_str(),
                                   &res.x, &res.y, &comp, STBI_rgb);
  // oof, rather implement OWL_RGB8.. :-)
  std::vector<unsigned char> texels(res.x*res.y*4);
  for (int y=res.y-1; y>=0; --y) {
    for (int x=0; x<res.x; ++x) {
      int index = (y*res.x+x)*4;
      texels[index]=*image++;
      texels[index+1]=*image++;
      texels[index+2]=*image++;
      texels[index+3]=(comp==3) ? 1U : *image++;
    }
  }
  OWLTexture ka_map
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         res.x,res.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);
  OWLTexture kd_map
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         res.x,res.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);

  OWLGeom parallelogramGeom = owlGeomCreate(context,parallelogramGeomType);
  owlGeomSetPrimCount(parallelogramGeom,1);
  owlGeomSet4f(parallelogramGeom,"plane",plane.x,plane.y,plane.z,plane.w);
  owlGeomSet3f(parallelogramGeom,"v1",v1.x,v1.y,v1.z);
  owlGeomSet3f(parallelogramGeom,"v2",v2.x,v2.y,v2.z);
  owlGeomSet3f(parallelogramGeom,"anchor",anchor.x,anchor.y,anchor.z);
  owlGeomSetTexture(parallelogramGeom,"ka_map",ka_map);
  owlGeomSetTexture(parallelogramGeom,"kd_map",kd_map);
  owlGeomSet3f(parallelogramGeom,"material.Ka",.35f,.35f,.35f);
  owlGeomSet3f(parallelogramGeom,"material.Kd",.5f,.5f,.5f);
  owlGeomSet3f(parallelogramGeom,"material.Ks",1.f,1.f,1.f);
  owlGeomSet3f(parallelogramGeom,"material.reflectivity",0.f,0.f,0.f);
  owlGeomSet1f(parallelogramGeom,"material.phong_exp",1.f);

  groups[1] = owlUserGeomGroupCreate(context,1,&parallelogramGeom);
  owlGroupBuildAccel(groups[1]);
  world
    = owlInstanceGroupCreate(context,2,
                             groups,
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
    { vec3f( -30.0f, -10.0f, 80.0f ), vec3f( 1.0f, 1.0f, 1.0f ), 1 },
    { vec3f(  10.0f,  30.0f, 20.0f ), vec3f( 1.0f, 1.0f, 1.0f ), 1 }
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
    // { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
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
  if (setup == SETUP_1984) {
  	// Jitter the location of the pool ball for motion blur
  	vec3f offset = random1() * vec3f(0.1f, 0.6f, 0.0f);
    std::vector<vec3f> center(poolballs.center);
    center[4] += offset;
    owlBufferUpload(poolballs.centerBuffer,center.data());
    owlGroupBuildAccel(groups[0]);
    owlGroupBuildAccel(world);
  }

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
  LOG("owl::ng example '" << av[0] << "' starting up");

  std::string arg1;
  if (ac>1) {
    arg1 = std::string(av[1]);
    if (arg1=="-h") {
      std::cout << "Usage: " << av[0] << "[-h|-1984]\n";
      std::cout << "  -h:    print this message\n";
      std::cout << "  -1984: load the 1984 motion blur scene\n";
      exit(EXIT_SUCCESS);
    }
  }

  Viewer viewer(arg1=="-1984"? Viewer::SETUP_1984: Viewer::SETUP_TRIANGLE);
  if (viewer.setup==Viewer::SETUP_TRIANGLE) {
    viewer.camera.setOrientation(cameraTriangle::init_lookFrom,
                                 cameraTriangle::init_lookAt,
                                 cameraTriangle::init_lookUp,
                                 owl::viewer::toDegrees(acosf(cameraTriangle::init_cosFovy)));
  } else {
    viewer.camera.setOrientation(camera1984::init_lookFrom,
                                 camera1984::init_lookAt,
                                 camera1984::init_lookUp,
                                 owl::viewer::toDegrees(acosf(camera1984::init_cosFovy)));
  }
  viewer.enableFlyMode();
  viewer.enableInspectMode(viewer::OWLViewer::Arcball,
                           owl::box3f(vec3f(-10.f),vec3f(+10.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
