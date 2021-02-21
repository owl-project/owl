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

// VOX file reader
#include "readVox.h"
#include "ogt_vox.h"

#include <cassert>

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

const int NUM_VERTICES = 8;
vec3f vertices[NUM_VERTICES] =
  {
    { -1.f,-1.f,-1.f },
    { +1.f,-1.f,-1.f },
    { -1.f,+1.f,-1.f },
    { +1.f,+1.f,-1.f },
    { -1.f,-1.f,+1.f },
    { +1.f,-1.f,+1.f },
    { -1.f,+1.f,+1.f },
    { +1.f,+1.f,+1.f }
  };

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
  {
    { 0,1,3 }, { 2,3,0 },
    { 5,7,6 }, { 5,6,4 },
    { 0,4,5 }, { 0,5,1 },
    { 2,3,7 }, { 2,7,6 },
    { 1,5,7 }, { 1,7,3 },
    { 4,0,2 }, { 4,2,6 }
  };

// const vec2i fbSize(800,600);
const vec3f init_lookFrom = vec3f(-4.f,+3.f,-2.f)*0.6f;
//const vec3f init_lookFrom(-4.f,+3.f,-2.f);
const vec3f init_lookAt(0.f,0.f,0.f);
const vec3f init_lookUp(0.f,1.f,0.f);
const float init_cosFovy = 0.66f;






struct Viewer : public owl::viewer::OWLViewer
{
  Viewer(const ogt_vox_scene *scene, const VoxelModel &model, uchar4 *palette);
  
  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;
  
      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */     
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;

  OWLGroup createInstancedTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, const VoxelModel &model, uchar4 *palette);
  OWLGroup createUserGeometryScene(OWLModule module, const ogt_vox_scene *scene);

  bool sbtDirty = true;
  OWLRayGen rayGen   { 0 };
  OWLContext context { 0 };
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
  sbtDirty = true;
}

// Adapted from ogt demo code. 
// The OGT format stores voxels as solid grids; we only need to store the nonempty entries on device.

std::vector<uchar4> extractSolidVoxelsFromModel(const ogt_vox_model* model)
{
  uint32_t solid_voxel_count = 0;
  uint32_t voxel_index = 0;
  std::vector<uchar4> solid_voxels;
  solid_voxels.reserve(model->size_z * model->size_y * model->size_x);
  for (uint32_t z = 0; z < model->size_z; z++) {
    for (uint32_t y = 0; y < model->size_y; y++) {
      for (uint32_t x = 0; x < model->size_x; x++, voxel_index++) {
        // if color index == 0, this voxel is empty, otherwise it is solid.
        uint8_t color_index = model->voxel_data[voxel_index];
        bool is_voxel_solid = (color_index != 0);
        // add to our accumulator
        solid_voxel_count += (is_voxel_solid ? 1 : 0);
        if (is_voxel_solid) {
          solid_voxels.push_back(
              // Switch to Y-up
              make_uchar4(uint8_t(x), uint8_t(y), uint8_t(z), color_index));
        }
      }
    }
  }
  LOG("solid voxel count: " << solid_voxel_count);
  return solid_voxels;
}

OWLGroup Viewer::createUserGeometryScene(OWLModule module, const ogt_vox_scene *scene)
{
  // -------------------------------------------------------
  // declare user vox geometry type
  // -------------------------------------------------------

  OWLVarDecl voxGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(VoxGeomData,prims)},
    { "colorPalette",  OWL_BUFPTR, OWL_OFFSETOF(VoxGeomData,colorPalette)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType voxGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(VoxGeomData),
                        voxGeomVars, -1);
  owlGeomTypeSetClosestHit(voxGeomType, 0, module, "VoxGeom");
  owlGeomTypeSetIntersectProg(voxGeomType, 0, module, "VoxGeom");
  owlGeomTypeSetBoundsProg(voxGeomType, module, "VoxGeom");

  // Do this before setting up user geometry, to compile bounds program
  owlBuildPrograms(context);

  LOG("building user geometries ...");

  // Only handle first model for now.  TODO: handle instances and multiple models
  assert(scene->num_models > 0);
  const ogt_vox_model *vox_model = scene->models[0];
  assert(vox_model);
  std::vector<uchar4> voxdata = extractSolidVoxelsFromModel(vox_model);

  // ------------------------------------------------------------------
  // VOX geom
  // ------------------------------------------------------------------
  OWLBuffer primBuffer 
    = owlDeviceBufferCreate(context, OWL_UCHAR4, voxdata.size(), voxdata.data());
  OWLBuffer paletteBuffer
    = owlDeviceBufferCreate(context, OWL_UCHAR4, 256, scene->palette.color);

  OWLGeom voxGeom = owlGeomCreate(context, voxGeomType);
  
  owlGeomSetPrimCount(voxGeom, voxdata.size());

  owlGeomSetBuffer(voxGeom, "prims", primBuffer);
  owlGeomSetBuffer(voxGeom, "colorPalette", paletteBuffer);


  // ------------------------------------------------------------------
  // bottom level group/accel
  // ------------------------------------------------------------------
  OWLGroup userGeomGroup
    = owlUserGeomGroupCreate(context,1,&voxGeom);
  owlGroupBuildAccel(userGeomGroup);


  // Normalize model using single instance transform
  const vec3f dims(float(vox_model->size_x), float(vox_model->size_y), float(vox_model->size_z));
  const float maxDim = owl::reduce_max(dims);
  const float worldScale = 2.0f / maxDim;  // 2 here to match our triangle-based box which lies in [-1,1]

  // object2world, read in right to left order:
  owl::affine3f transform = owl::affine3f::scale(worldScale) *  // normalize 
    owl::affine3f::rotate(vec3f(1,0,0), -0.5f*M_PI) *           // rotate Z-up to Y-up
    owl::affine3f::translate(-dims*0.5f);                       // center about origin

  OWLGroup world = owlInstanceGroupCreate(context, 1);
  owlInstanceGroupSetChild(world, 0, userGeomGroup);
  owlInstanceGroupSetTransform(world, 0, (const float*)&transform, OWL_MATRIX_FORMAT_OWL); 

  return world;

}

OWLGroup Viewer::createInstancedTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, const VoxelModel &model, uchar4 *palette)
{
  // -------------------------------------------------------
  // declare triangle-based box geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "colorPerInstance",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colorPerInstance)}
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,3);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

  LOG("building triangle geometries ...");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,NUM_VERTICES,vertices);
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,NUM_INDICES,indices);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_INDICES,sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  const int numInstances = model.voxels.size();
  std::vector<owl::vec3f> colors;
  colors.reserve(numInstances);
  for (int i = 0; i < numInstances; ++i) {
      uchar4 col = palette[model.voxels[i].w];
      colors.push_back({ col.x / 255.0f, col.y / 255.0f, col.z / 255.0f });
  }
  OWLBuffer colorBuffer
      = owlDeviceBufferCreate(context, OWL_FLOAT3, numInstances, colors.data());
  owlGeomSetBuffer(trianglesGeom, "colorPerInstance", colorBuffer);
  
  // ------------------------------------------------------------------
  // the group/accel for that mesh
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);

  // ------------------------------------------------------------------
  // instances
  // ------------------------------------------------------------------

  std::vector<owl::affine3f> transforms;
  transforms.reserve(numInstances);
  
  // Normalize model
  const owl::vec3f dims(float(model.dims[0]), float(model.dims[1]), float(model.dims[2]));
  const float maxDim = owl::reduce_max(dims);
  const float worldScale = 1.0f / maxDim;

  for (size_t i = 0; i < numInstances; ++i) {
      uchar4 b = model.voxels[i];
      // Note: some unintuitive transforms here to account for our modeled box being 2 units
      // long on each side, and centered about the origin.
      owl::vec3f trans = owl::vec3f(b.x, b.y, b.z)*2.0f - dims + owl::vec3f(1.0f);
      transforms.push_back(owl::affine3f::scale(worldScale) * owl::affine3f::translate(trans));
  }

  OWLGroup world = owlInstanceGroupCreate(context, transforms.size());

  for (int i = 0; i < int(transforms.size()); ++i) {
    owlInstanceGroupSetChild(world, i, trianglesGroup);  // All instances point to the same geometry
    owlInstanceGroupSetTransform(world, i, (const float*)&transforms[i], OWL_MATRIX_FORMAT_OWL);
  }

  return world;
  
}

Viewer::Viewer(const ogt_vox_scene *scene, const VoxelModel &model, uchar4 *palette)
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  OWLModule module = owlModuleCreate(context,ptxCode);

  
  OWLGroup world = createInstancedTriangleGeometryScene(module, scene, model, palette);
  //OWLGroup world = createUserGeometryScene(module, scene);

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
  
  // Build all programs again, even if they have been built already for bounds program
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);
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

  if (ac < 2) {
      LOG("missing expected argument for .vox file. Exiting");
      exit(1);
  }
  const std::string infile(av[1]);
  std::vector<VoxelModel> models;
  uchar4 palette[256];
  const ogt_vox_scene *scene = nullptr;
  try {
      readVox( infile.c_str(), models, palette );
      scene = loadVoxSceneOGT(infile.c_str());

  } catch ( const std::exception& e ) {
      LOG("Caught exception while reading voxel model: " << infile);
      LOG(e.what());
      exit(1);
  }

  if (!scene) {
      LOG("Could not read vox file: " << infile.c_str() << ", exiting");
      exit(1);
  }
  if (scene->num_models == 0) {
      LOG("No voxel models found in file, exiting");
      ogt_vox_destroy_scene(scene);
      exit(1);
  }

  if ( models.empty() ) {
      LOG("No voxels found in file, exiting");
      exit(1);
  }

  Viewer viewer(scene, models[0], palette);
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

  ogt_vox_destroy_scene(scene);
}
