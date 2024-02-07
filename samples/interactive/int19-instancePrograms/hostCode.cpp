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

const vec3i numBoxes(100);
const uint32_t numBoxTypes = 100; 
const float worldSize = 1;
const vec3f boxSize   = (2*.4f*worldSize)/vec3f(numBoxes);
const float animSpeed = 4.f;

std::default_random_engine rndGen;
std::uniform_real_distribution<float> distribution_uniform(0.0,1.0);
std::uniform_real_distribution<float> distribution_speed(.1f,.8f);
std::uniform_int_distribution<int> distribution_texSize(2,16);

OWLParams lp;
OWLBuffer BLASBuffer;
OWLBuffer BLASOffsetsBuffer;

struct BoxAnimState {
  void init(vec3i boxID)
  {
    vec3f rel = (vec3f(boxID)+.5f) / vec3f(numBoxes);
    boxCenter = vec3f(-worldSize) + (2.f*worldSize)*rel;
    rotationAngle0 = float(distribution_uniform(rndGen)*(2.f*M_PI));
    do {
      rotationAxis.x = distribution_uniform(rndGen);
      rotationAxis.y = distribution_uniform(rndGen);
      rotationAxis.z = distribution_uniform(rndGen);
    } while (dot(rotationAxis,rotationAxis) > 1.f);
    rotationAxis = normalize(rotationAxis);
    rotationSpeed = distribution_speed(rndGen);
  }

  affine3f getTransform(float t) const
  {
    const float angle  = rotationAngle0 + rotationSpeed*t;
    const linear3f rot = linear3f::rotate(rotationAxis,angle);
    return affine3f(rot,boxCenter);
  }

  vec3f boxCenter;
  vec3f rotationAxis;
  float rotationSpeed;
  float rotationAngle0;
};

std::vector<BoxAnimState> boxAnimStates;
std::vector<affine3f>     boxTransforms;

void addFace(Mesh &mesh, const vec3f ll, const vec3f du, const vec3f dv)
{
  int idxll = (int)mesh.vertices.size();
  for (int iy=0;iy<2;iy++)
    for (int ix=0;ix<2;ix++) {
      mesh.vertices.push_back(ll+float(ix)*du+float(iy)*dv);
      mesh.texCoords.push_back(vec2f((float)ix, (float)iy));
    }
  mesh.indices.push_back(vec3i(idxll,idxll+1,idxll+3));
  mesh.indices.push_back(vec3i(idxll,idxll+3,idxll+2));
}

void addBox(Mesh &mesh,
            const vec3f du=vec3f(boxSize.x,0,0),
            const vec3f dv=vec3f(0,boxSize.y,0),
            const vec3f dw=vec3f(0,0,boxSize.z))
{
  addFace(mesh,-0.5f*(du+dv+dw),du,dv);
  addFace(mesh,-0.5f*(du+dv+dw),du,dw);
  addFace(mesh,-0.5f*(du+dv+dw),dv,dw);

  addFace(mesh,0.5f*(du+dv+dw),-du,-dv);
  addFace(mesh,0.5f*(du+dv+dw),-du,-dw);
  addFace(mesh,0.5f*(du+dv+dw),-dv,-dw);
}

OWLGroup createBox(OWLContext context,
                 OWLGeomType trianglesGeomType,
                 const vec3i coord)
{
  Mesh mesh;
  addBox(mesh);

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertices.size(),mesh.vertices.data());
  OWLBuffer texCoordsBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT2,mesh.texCoords.size(),mesh.texCoords.data());
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,mesh.indices.size(),mesh.indices.data());
  // OWLBuffer frameBuffer
  //   = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          mesh.vertices.size(),sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         mesh.indices.size(),sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"texCoord",texCoordsBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  // ------------------------------------------------------------------
  // create a 4x4 checkerboard texture
  // ------------------------------------------------------------------
  vec2i texSize(distribution_texSize(rndGen),distribution_texSize(rndGen));
  vec4uc color0 = vec4uc(255.99f*vec4f(
      (float)distribution_uniform(rndGen),
      (float)distribution_uniform(rndGen),
      (float)distribution_uniform(rndGen),
      0.f));
  vec4uc color1 = vec4uc(255)-color0;
  std::vector<vec4uc> texels;
  for (int iy=0;iy<texSize.y;iy++)
    for (int ix=0;ix<texSize.x;ix++) {
      texels.push_back(((ix ^ iy)&1) ?
                       color0 : color1);
    }
  OWLTexture cbTexture
    = owlTexture2DCreate(context,
                         OWL_TEXEL_FORMAT_RGBA8,
                         texSize.x,texSize.y,
                         texels.data(),
                         OWL_TEXTURE_NEAREST,
                         OWL_TEXTURE_CLAMP);
  owlGeomSetTexture(trianglesGeom,"texture",cbTexture);

  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);

  owlGroupBuildAccel(trianglesGroup);

  return trianglesGroup;
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
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // disable the geometry from contribution to allow for device-side 
  // instance manipulation. In this example, our own group only has one geometry, 
  // so this technically has no effect. But more generally, we're making an agreement 
  // with OWL that _if_ we were to give more than one geometry in a triangles group, 
  // that it's _okay_ for OWL to just make one hit group that's shared across all 
  // geometries in that group. 
  //
  // It's a subtle difference, but the end result is that the sbtOffset for every geometry 
  // group in our instance accel can be calculated as: numRayTypes * blasIndex
  // which is _waaaay_ easier to generate on the device in a CUDA kernel. 
  owlContextDisablePerGeometrySBTRecords(context);

  // -------------------------------------------------------
  // set up launch params
  // -------------------------------------------------------
  OWLVarDecl lpVars[] = {
    { "numBoxes",     OWL_UINT3,   OWL_OFFSETOF(Globals,numBoxes)},
    { "time",         OWL_FLOAT,   OWL_OFFSETOF(Globals,time)},
    { "BLAS",         OWL_BUFPTR,  OWL_OFFSETOF(Globals,BLAS)},
    { "BLASOffsets",  OWL_BUFPTR,  OWL_OFFSETOF(Globals,BLASOffsets)},
    { "numBLAS",      OWL_UINT,    OWL_OFFSETOF(Globals,numBLAS)},
    { nullptr /* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  lp = owlParamsCreate(context,
                      sizeof(Globals),
                      lpVars,-1);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texCoord)},
    { "texture",  OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData,texture)},
    { nullptr }
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

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
    { /* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);
  
  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // Three random boxes
  std::vector<uint32_t> BLASOffsets;
  std::vector<OptixTraversableHandle> BLASes;
  for (int i = 0; i < numBoxTypes; ++i) {
    OWLGroup box = createBox(context,trianglesGeomType,vec3i(i,0,0));
    BLASes.push_back(owlGroupGetTraversable(box, 0));
    BLASOffsets.push_back(owlGroupGetSBTOffset(box));
  }
 
  BLASBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(uint32_t), BLASOffsets.size(), BLASOffsets.data());
  BLASOffsetsBuffer = owlDeviceBufferCreate(context, OWL_USER_TYPE(OptixTraversableHandle), BLASes.size(), BLASes.data());

  owlParamsSet3ui(lp, "numBoxes", numBoxes.x, numBoxes.y, numBoxes.z);
  owlParamsSetBuffer(lp, "BLAS", BLASOffsetsBuffer);
  owlParamsSetBuffer(lp, "BLASOffsets", BLASBuffer);
  owlParamsSet1ui(lp, "numBLAS", BLASes.size());
  owlParamsSet1f(lp, "time", 42.f);

  world = owlInstanceGroupCreate(context,numBoxes.x * numBoxes.y * numBoxes.z,
                            nullptr, nullptr, nullptr,
                            OWL_MATRIX_FORMAT_OWL,
                            OPTIX_BUILD_FLAG_PREFER_FAST_TRACE | OPTIX_BUILD_FLAG_ALLOW_UPDATE,
                             /* use instance program */ true);
  
  owlInstanceGroupSetInstanceProg(world,module,"instanceProg");

  // Build programs
  owlBuildPrograms(context);

  // Build our instance accel on the device
  owlGroupBuildAccel(world, lp);
  
  /* camera and frame buffer get set in resiez() and cameraChanged() */
  owlRayGenSetGroup (rayGen,"world",        world);
  
  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  owlBuildPipeline(context);
  owlBuildSBT(context);
}

void Viewer::render()
{
  static double t0 = getCurrentTime();
  double t = animSpeed * (getCurrentTime() - t0);
  owlParamsSet1f(lp,"time",(float)t);

  static double updateTime = 0.f;
  updateTime -= getCurrentTime();
  static int frameID = 0;
  frameID++;
  // we can resort to update here because the initial build was
  // already done before
  owlGroupRefitAccel(world,lp);
  updateTime += getCurrentTime();
  PRINT(updateTime/frameID);

  owlRayGenSetGroup(rayGen,"world",world);

  // owlGroupBuildAccel(world);
  owlBuildSBT(context);
  owlLaunch2D(rayGen,fbSize.x,fbSize.y,lp);
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
