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
#include "owl/helper/cuda.h"
// viewer base class, for window and user interaction
#include "owlViewer/OWLViewer.h"
#include "owl/common/math/AffineSpace.h"
#include <random>
#include <fstream>

using namespace owl::common;

int maxInstances = 1<<30;

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
  std::vector<vec3i> indices;
};

OWLGroup createBox(OWLContext context,
                   OWLGeomType trianglesGeomType,
                   const box3f &box,
                   int objID)
{
  Mesh mesh;
  for (int iz=0;iz<2;iz++)
    for (int iy=0;iy<2;iy++)
      for (int ix=0;ix<2;ix++) {
        mesh.vertices.push_back(vec3f((&box.lower)[ix].x,
                                      (&box.lower)[iy].y,
                                      (&box.lower)[iz].z));
      }
  mesh.indices = std::vector<vec3i>
    {
     { 0,1,3 }, { 2,3,0 },
     { 5,7,6 }, { 5,6,4 },
     { 0,4,5 }, { 0,5,1 },
     { 2,3,7 }, { 2,7,6 },
     { 1,5,7 }, { 1,7,3 },
     { 4,0,2 }, { 4,2,6 }
    };
  
  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertices.size(),mesh.vertices.data());
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
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  vec3f color = owl::randomColor(objID);
  owlGeomSet3f(trianglesGeom,"color",color.x,color.y,color.z);

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
  Viewer(const std::string &inFileName);

  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;

      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  void loadModel(const std::string &inFileName);

  std::vector<box3f>    objBoxes;
  std::vector<OWLGroup> objGroup;
  std::vector<int>      inst_objIDs;
  std::vector<affine3f> inst_xfms;
  box3f worldBounds;
  
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

void Viewer::loadModel(const std::string &inFileName)
{
  std::ifstream in(inFileName);
  if (!in.good()) throw std::runtime_error("could not open " + inFileName);

  int numObjects;
  std::string skip;
  in >> skip >> skip >> numObjects;
  if (!in.good()) throw std::runtime_error("error reading " + inFileName + " (1)");
  PRINT(numObjects);
  objBoxes.resize(numObjects);
  for (auto &obj : objBoxes)
    in
      >> obj.lower.x
      >> obj.lower.y
      >> obj.lower.z
      >> obj.upper.x
      >> obj.upper.y
      >> obj.upper.z;

  if (!in.good()) throw std::runtime_error("error reading " + inFileName + " (2)");

  int numInstances;
  in >> skip >> skip >> numInstances;


  numInstances = std::min(numInstances,maxInstances);
  
  if (!in.good()) throw std::runtime_error("error reading " + inFileName + " (3)");
  PRINT(numInstances);
  inst_objIDs.resize(numInstances);
  inst_xfms.resize(numInstances);
  for (int i=0;i<numInstances;i++) {
    in >> inst_objIDs[i];
    in
      >> inst_xfms[i].l.vx.x
      >> inst_xfms[i].l.vx.y
      >> inst_xfms[i].l.vx.z
      >> inst_xfms[i].l.vy.x
      >> inst_xfms[i].l.vy.y
      >> inst_xfms[i].l.vy.z
      >> inst_xfms[i].l.vz.x
      >> inst_xfms[i].l.vz.y
      >> inst_xfms[i].l.vz.z
      >> inst_xfms[i].p.x
      >> inst_xfms[i].p.y
      >> inst_xfms[i].p.z;

    for (int iz=0;iz<2;iz++)
      for (int iy=0;iy<2;iy++)
        for (int ix=0;ix<2;ix++) {
          box3f box = objBoxes[inst_objIDs[i]];
          vec3f corner((&box.lower)[ix].x,
                       (&box.lower)[ix].y,
                       (&box.lower)[ix].z);
          worldBounds.extend(xfmPoint(inst_xfms[i],corner));
        }
  }
  if (!in.good()) throw std::runtime_error("error reading " + inFileName + " (4)");
  std::cout << "scene read ... " << std::endl;
  PRINT(worldBounds);
}

Viewer::Viewer(const std::string &inFileName)
{
  loadModel(inFileName);
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  OWLModule module = owlModuleCreate(context,deviceCode_ptx);

  // ##################################################################
  // set up all the *GEOMETRY* graph we want to render
  // ##################################################################

  // -------------------------------------------------------
  // declare geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "color", OWL_FLOAT3, OWL_OFFSETOF(TrianglesGeomData,color)},
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
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // addCube(vec3f(0.f),
  //         vec3f(2.f,0.f,0.f),
  //         vec3f(0.f,2.f,0.f),
  //         vec3f(0.f,0.f,2.f));

  // one group per object
  std::vector<OWLGroup> objGroups;
  for (int i=0;i<objBoxes.size();i++)
    objGroups.push_back(createBox(context,trianglesGeomType,objBoxes[i],i));

  // one (object-)group handle per instance
  std::vector<OWLGroup> inst_groups;
  for (int instID=0;instID < inst_objIDs.size();instID++) {
    int objID = inst_objIDs[instID];
    if (objID < 0 || objID >= objGroups.size())
      throw std::runtime_error("invalid object id .. "+std::to_string(objID));
    inst_groups.push_back(objGroups[objID]);
  }

  PRINT(inst_groups.size());
  PRINT(inst_xfms.size());
  world
    = owlInstanceGroupCreate(context,
                             inst_groups.size(),
                             inst_groups.data(),
                             nullptr,
                             (const float*)inst_xfms.data(),
                             OWL_MATRIX_FORMAT_OWL,
                             OPTIX_BUILD_FLAG_PREFER_FAST_TRACE
                             | OPTIX_BUILD_FLAG_ALLOW_UPDATE);
  owlGroupBuildAccel(world);
  
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
  /* camera and frame buffer get set in resiez() and cameraChanged() */
  owlRayGenSetGroup (rayGen,"world",        world);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################

  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);
}

void Viewer::render()
{
  if (sbtDirty) { owlBuildSBT(context); sbtDirty = false; }

  // static double t0 = getCurrentTime();
  // double t = animSpeed * (getCurrentTime() - t0);
  // for (size_t i=0;i<boxTransforms.size();i++) {
  //   boxTransforms[i] = boxAnimStates[i].getTransform((float)t);
  //   owlInstanceGroupSetTransform(world,(int)i,
  //                                (const float*)&boxTransforms[i],
  //                                OWL_MATRIX_FORMAT_OWL);
  // }

  static double updateTime = 0.f;
  updateTime -= getCurrentTime();
  static int frameID = 0;
  frameID++;
  // we can resort to update here because the initial build was
  // already done before
  // owlGroupRefitAccel(world);
  updateTime += getCurrentTime();

  // owlGroupBuildAccel(world);
  // owlBuildSBT(context);

  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
}


int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");
  std::string inFileName = "";
  bool haveCamera = false;
  vec3f vp, vi, vu;
  float fovy;
  
  for (int i=1;i<ac;i++) {
    const std::string arg = av[i];
    if (arg[0] != '-')
      inFileName = arg;
    else if (arg == "-mi" || arg == "--max-instances") {
      maxInstances = std::stoi(av[++i]);
    } else if (arg == "--camera") {
      haveCamera = true;
      vp.x = std::atof(av[++i]);
      vp.y = std::atof(av[++i]);
      vp.z = std::atof(av[++i]);
      vi.x = std::atof(av[++i]);
      vi.y = std::atof(av[++i]);
      vi.z = std::atof(av[++i]);
      vu.x = std::atof(av[++i]);
      vu.y = std::atof(av[++i]);
      vu.z = std::atof(av[++i]);
      fovy = std::atof(av[++i]);
    } else
      throw std::runtime_error("unknown cmdline arg "+arg);
  }
  if (inFileName.empty())
    throw std::runtime_error("./int16_instanceViewer inFile.instances");
  Viewer viewer(av[1]);
  if (!haveCamera)
    viewer.camera.setOrientation(viewer.worldBounds.center()-viewer.worldBounds.size(),
                                 viewer.worldBounds.center(),
                                 vec3f(0.f,1.f,0.f),50.f);
  else 
    viewer.camera.setOrientation(vp,vi,vu,fovy);
  // vec3f(-13508.9f, 9494.87f, 2172.44f),
  //                             vec3f(138619.f, 1344.35f, -73468.3f),
  //                             vec3f(0.f, 1.f, 0.f),
  //                             50.f);

  
  viewer.setWorldScale(length(viewer.worldBounds.size()));
  viewer.enableFlyMode();
  viewer.enableInspectMode(viewer.worldBounds);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  viewer.showAndRun();
}
