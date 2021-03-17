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

#include "constants.h"
#include "ogt_vox.h"
#include "readVox.h"

#include <cassert>
#include <map>
#include <tuple>
#include <vector>

// annoying stuff so we can list files in a directory later
#if __cplusplus < 201703L   // i.e. not c++17
#define _SILENCE_EXPERIMENTAL_FILESYSTEM_DEPRECATION_WARNING
#include <experimental/filesystem>
namespace std {
  namespace filesystem = experimental::filesystem;
}
#else
#include <filesystem>
using filesystem = std::filesystem
#endif

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];


const vec3f init_lookAt {0.0f};
const float init_cosFovy = 0.10f;  // small fov to approach isometric
const vec3f init_lookUp(0.f, 0.f, 1.f);

const float isometricAngle = 35.564f * M_PIf/180.0f;
const owl::affine3f cameraRotation = 
  owl::affine3f::rotate(vec3f(0,0,1), M_PIf/4.0f) *
  owl::affine3f::rotate(vec3f(-1,0,0), isometricAngle);

const vec3f init_lookFrom = xfmPoint(cameraRotation, vec3f(0, -30.f, 0));


enum SceneType {
  SCENE_TYPE_FLAT=1,
  SCENE_TYPE_USER,
  SCENE_TYPE_INSTANCED,
  SCENE_TYPE_INVALID
};

struct Viewer : public owl::viewer::OWLViewer
{

  struct GlobalOptions {
    bool enableGround = true;
    bool enableClipping = true;
    bool enableToonOutline = true;
    bool enableCulling = false;
  };

  Viewer(const ogt_vox_scene *scene, SceneType sceneType, const GlobalOptions &options);
  
  /*! gets called whenever the viewer needs us to re-render out widget */
  void render() override;
  
      /*! window notifies us that we got resized. We HAVE to override
          this to know our actual render dimensions, and get pointer
          to the device frame buffer that the viewer cated for us */     
  void resize(const vec2i &newSize) override;

  /*! this function gets called whenever any camera manipulator
    updates the camera. gets called AFTER all values have been updated */
  void cameraChanged() override;


  void key(char key, const vec2i &/*where*/) override;

  /// Geometry creation functions

  OWLGroup createFlatTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
      owl::box3f &sceneBox);
  OWLGroup createInstancedTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
      owl::box3f &sceneBox);
  OWLGroup createUserGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
      owl::box3f &sceneBox);


  bool sbtDirty = true;
  OWLRayGen rayGen   { 0 };
  OWLLaunchParams launchParams { 0 };
  OWLContext context { 0 };
  int frameID = 0;
  OWLBuffer fbAccum = nullptr;

  owl::box3f sceneBox;  // in units of bricks
  int clipHeight = 1000;
  bool clipDirty = false;
  float sunPhi   = 0.785398f;  // rotation about up axis
  float sunTheta = 0.785398f;  // elevation angle, 0 at horizon
  bool sunDirty = true;
  bool enableGround = true;
  bool enableClipping = true;  // enable clipping plane in shader
  bool enableToonOutline = true;
  bool enableCulling = false;  // cull hidden bricks
  
};

/*! window notifies us that we got resized */     
void Viewer::resize(const vec2i &newSize)
{
  OWLViewer::resize(newSize);

  if (fbAccum) {
    owlBufferDestroy(fbAccum);
  }
  fbAccum = owlDeviceBufferCreate(context, OWL_FLOAT4, newSize.x*newSize.y, nullptr);
  owlParamsSetBuffer(launchParams, "fbAccumBuffer", fbAccum);
  cameraChanged();
}

/*! window notifies us that the camera has changed */
void Viewer::cameraChanged()
{
  frameID = 0; // reset accum
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
  owlParamsSet1ul(launchParams, "fbPtr",  (uint64_t)fbPointer);
  owlParamsSet2i (launchParams, "fbSize", (const owl2i&)fbSize);

  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);
  sbtDirty = true;
}


void Viewer::key(char key, const vec2i &where)
{
  switch (key) {
    case '1':
      sunTheta += 0.1f;
      sunTheta = owl::clamp(sunTheta, 0.f, M_PIf/2);
      sunDirty = true;
      return;
    case '2':
      sunTheta -= 0.1f;
      sunTheta = owl::clamp(sunTheta, 0.f, M_PIf/2);
      sunDirty = true;
      return;
    case '3':
      sunPhi += 0.1f;
      if (sunPhi >= 2.0f*M_PIf) sunPhi -= 2.0f*M_PIf;
      sunDirty = true;
      return;
    case '4':
      sunPhi -= 0.1f;
      if (sunPhi <= 0.0f) sunPhi += 2.0f*M_PIf;
      sunDirty = true;
      return;
    case '[':
      clipHeight -= 1;
      if (clipHeight < 0) clipHeight = 0;
      clipDirty = true;
      return;
    case ']':
      clipHeight += 1;
      if (clipHeight > int(sceneBox.span().z)+1) {
        clipHeight = int(sceneBox.span().z)+1;
      }
      clipDirty = true;
      return;

    case 'p':
      const std::string filename = "screenshot.png";
      LOG("Saving screenshot to: " << filename);
      screenShot(filename);
      break;
  }
  OWLViewer::key(key, where);
}

// Adapted from ogt demo code. 
// The OGT format stores voxels as solid grids; we only need to store the nonempty entries on device.

std::vector<uchar4> extractSolidVoxelsFromModel(const ogt_vox_model* model, bool cullHidden)
{
  // is the voxel at the given index opaque?
  auto isOpaque = [model](int x, int y, int z) -> bool {
    if (x < 0 || x >= (int)model->size_x) return false;
    if (y < 0 || y >= (int)model->size_y) return false;
    if (z < 0 || z >= (int)model->size_z) return false;
    int index = z*model->size_x*model->size_y + y*model->size_x + x;
    uint8_t ci = model->voxel_data[index];
    return ci != 0;
  };

  uint32_t voxel_index = 0;
  std::vector<uchar4> solid_voxels;
  solid_voxels.reserve(model->size_z * model->size_y * model->size_x);
  for (int z = 0; z < (int)model->size_z; z++) {
    for (int y = 0; y < (int)model->size_y; y++) {
      for (int x = 0; x < (int)model->size_x; x++, voxel_index++) {
        uint8_t ci = model->voxel_data[voxel_index];
        if (ci) {
          if (cullHidden) {
            if (isOpaque(x+1, y, z) && isOpaque(x-1, y, z) &&
                isOpaque(x, y+1, z) && isOpaque(x, y-1, z) &&
                isOpaque(x, y, z+1) && isOpaque(x, y, z-1)) 
            {
              // all 6 face-neighbors of the brick are visible, so this brick is hidden
              continue;
            }
          }
          solid_voxels.push_back(
              // Switch to Y-up
              make_uchar4(uint8_t(x), uint8_t(y), uint8_t(z), ci));
        }
      }
    }
  }
  LOG("solid voxel count: " << solid_voxels.size());
  return solid_voxels;
}

// Simple memory tracker
struct BufferAllocator {
  inline OWLBuffer deviceBufferCreate(OWLContext  context,
                        OWLDataType type,
                        size_t      count,
                        const void *init)
  {
    OWLBuffer buf = owlDeviceBufferCreate(context, type, count, init);
    bytesAllocated += owlBufferSizeInBytes(buf);
    return buf;
  }

  size_t bytesAllocated = 0u;
};

void logSceneMemory(size_t bottomLevelBvhSizeInBytes,
                    size_t topLevelBvhSizeInBytes,
                    const BufferAllocator &allocator)
{
  LOG("Scene GAS memory: " << owl::common::prettyNumber(bottomLevelBvhSizeInBytes));
  LOG("Scene IAS memory: " << owl::common::prettyNumber(topLevelBvhSizeInBytes));
  LOG("Scene buffer memory: " << owl::common::prettyNumber(allocator.bytesAllocated));

  size_t total = bottomLevelBvhSizeInBytes + topLevelBvhSizeInBytes + allocator.bytesAllocated;
  LOG("Scene total memory: "  << total << " (" << owl::common::prettyNumber(total) << ")");
}

size_t getAccelSizeInBytes(OWLGroup group)
{
  size_t finalSize = 0u;
  owlGroupGetAccelSize(group, &finalSize, nullptr);
  return finalSize;
}


OWLGroup Viewer::createUserGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
    box3f &sceneBox) 
{
  BufferAllocator allocator;

  // -------------------------------------------------------
  // declare user vox geometry type
  // -------------------------------------------------------

  OWLVarDecl voxGeomVars[] = {
    { "prims",  OWL_BUFPTR, OWL_OFFSETOF(VoxGeomData,prims)},
    { "colorPalette",  OWL_BUFPTR, OWL_OFFSETOF(VoxGeomData,colorPalette)},
    { "enableToonOutline", OWL_BOOL, OWL_OFFSETOF(VoxGeomData, enableToonOutline)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType voxGeomType
    = owlGeomTypeCreate(context,
                        OWL_GEOMETRY_USER,
                        sizeof(VoxGeomData),
                        voxGeomVars, -1);
  owlGeomTypeSetClosestHit(voxGeomType, 0, module, "VoxGeom");

  owlGeomTypeSetIntersectProg(voxGeomType, 0, module, "VoxGeomMajercik");

  owlGeomTypeSetIntersectProg(voxGeomType, 1, module, "VoxGeomShadow");  // for shadow rays
  if (this->enableToonOutline) {
    owlGeomTypeSetIntersectProg(voxGeomType, 2, module, "VoxGeomOutlineShadow");
  }
  owlGeomTypeSetBoundsProg(voxGeomType, module, "VoxGeom");

  // Do this before setting up user geometry, to compile bounds program
  owlBuildPrograms(context);

  assert(scene->num_instances > 0);
  assert(scene->num_models > 0);


  // Palette buffer is global to scene
  OWLBuffer paletteBuffer
    = allocator.deviceBufferCreate(context, OWL_UCHAR4, 256, scene->palette.color);

  // Cluster instances together that use the same model
  std::map<uint32_t, std::vector<uint32_t>> modelToInstances;

  for (uint32_t instanceIndex = 0; instanceIndex < scene->num_instances; instanceIndex++) {
    const ogt_vox_instance &vox_instance = scene->instances[instanceIndex];
    modelToInstances[vox_instance.model_index].push_back(instanceIndex);
  }

  std::vector<OWLGroup> geomGroups;
  geomGroups.reserve(scene->num_models);

  std::vector<owl::affine3f> instanceTransforms;
  instanceTransforms.reserve(scene->num_instances);

  size_t totalSolidVoxelCount = 0;
  size_t bottomLevelBvhSizeInBytes = 0;
  
  // Make instance transforms
  for (auto it : modelToInstances) {

    const ogt_vox_model *vox_model = scene->models[it.first];
    assert(vox_model);
    std::vector<uchar4> voxdata = extractSolidVoxelsFromModel(vox_model, enableCulling);

    LOG("building user geometry for model ...");

    // ------------------------------------------------------------------
    // set up user primitives for single vox model
    // ------------------------------------------------------------------
    OWLBuffer primBuffer 
      = allocator.deviceBufferCreate(context, OWL_UCHAR4, voxdata.size(), voxdata.data());

    OWLGeom voxGeom = owlGeomCreate(context, voxGeomType);
    
    owlGeomSetPrimCount(voxGeom, voxdata.size());

    owlGeomSetBuffer(voxGeom, "prims", primBuffer);
    owlGeomSetBuffer(voxGeom, "colorPalette", paletteBuffer);
    owlGeomSet1b(voxGeom, "enableToonOutline", this->enableToonOutline);

    // ------------------------------------------------------------------
    // bottom level group/accel
    // ------------------------------------------------------------------
    OWLGroup userGeomGroup = owlUserGeomGroupCreate(context,1,&voxGeom);
    owlGroupBuildAccel(userGeomGroup);
    bottomLevelBvhSizeInBytes += getAccelSizeInBytes(userGeomGroup);

    LOG("adding (" << it.second.size() << ") instance transforms for model ...");
    for (uint32_t instanceIndex : it.second) {

      totalSolidVoxelCount += voxdata.size();

      const ogt_vox_instance &vox_instance = scene->instances[instanceIndex];

      const std::string instanceName = vox_instance.name ? vox_instance.name : "(unnamed)";
      if (vox_instance.hidden) {
        LOG("skipping hidden VOX instance: " << instanceName );
        continue;
      } else {
        LOG("building VOX instance: " << instanceName);
      }

      // VOX instance transform
      const ogt_vox_transform &vox_transform = vox_instance.transform;

      const owl::affine3f instanceTransform( 
          vec3f( vox_transform.m00, vox_transform.m01, vox_transform.m02 ),  // column 0
          vec3f( vox_transform.m10, vox_transform.m11, vox_transform.m12 ),  //  1
          vec3f( vox_transform.m20, vox_transform.m21, vox_transform.m22 ),  //  2
          vec3f( vox_transform.m30, vox_transform.m31, vox_transform.m32 )); //  3 

      // This matrix translates model to its center (in integer coords!) and applies instance transform
      const affine3f instanceMoveToCenterAndTransform = 
        affine3f(instanceTransform) * 
        affine3f::translate(-vec3f(float(vox_model->size_x/2),   // Note: snapping to int to match MV
                                   float(vox_model->size_y/2), 
                                   float(vox_model->size_z/2)));

      sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform, vec3f(0.0f)));
      sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform,  
            vec3f(float(vox_model->size_x), float(vox_model->size_y), float(vox_model->size_z))));

      instanceTransforms.push_back(instanceMoveToCenterAndTransform);

      geomGroups.push_back(userGeomGroup);

    }

  }

  LOG("Total solid voxels in all instanced models: " << totalSolidVoxelCount);

  const vec3f sceneCenter = sceneBox.center();
  const vec3f sceneSpan = sceneBox.span();

  if (this->enableGround) {
    // ------------------------------------------------------------------
    // set up vox data and accels for ground (single stretched brick)
    // ------------------------------------------------------------------
    
    std::vector<uchar4> voxdata {make_uchar4(0,0,0, 249)};  // using color index of grey in default palette
    OWLBuffer primBuffer 
      = allocator.deviceBufferCreate(context, OWL_UCHAR4, voxdata.size(), voxdata.data());

    OWLGeom voxGeom = owlGeomCreate(context, voxGeomType);
    
    owlGeomSetPrimCount(voxGeom, voxdata.size());

    owlGeomSetBuffer(voxGeom, "prims", primBuffer);
    owlGeomSetBuffer(voxGeom, "colorPalette", paletteBuffer);
    owlGeomSet1b(voxGeom, "enableToonOutline", false);  // no outline for ground because it's scaled

    OWLGroup userGeomGroup = owlUserGeomGroupCreate(context,1,&voxGeom);
    owlGroupBuildAccel(userGeomGroup);
    bottomLevelBvhSizeInBytes += getAccelSizeInBytes(userGeomGroup);
    geomGroups.push_back(userGeomGroup);

    instanceTransforms.push_back( 
        owl::affine3f::translate(vec3f(sceneCenter.x, sceneCenter.y, sceneCenter.z - 1 - 0.5f*sceneSpan.z)) *
        owl::affine3f::scale(vec3f(2*sceneSpan.x, 2*sceneSpan.y, 1.0f))*    // assume Z up
        owl::affine3f::translate(vec3f(-0.5f, -0.5f, 0.0f))
        );
  }

  // Apply final scene transform so we can use the same camera for every scene
  const float maxSpan = owl::reduce_max(sceneBox.span());
  owl::affine3f worldTransform = 
    owl::affine3f::scale(2.0f/maxSpan) *                                    // normalize
    owl::affine3f::translate(vec3f(-sceneCenter.x, -sceneCenter.y, -sceneBox.lower.z));  // center about (x,y) origin ,with Z up to match MV

  for (size_t i = 0; i < instanceTransforms.size(); ++i) {
    instanceTransforms[i] = worldTransform * instanceTransforms[i];
  }
  
  OWLGroup world = owlInstanceGroupCreate(context, instanceTransforms.size(),
      geomGroups.data(), nullptr, (const float*)instanceTransforms.data(), OWL_MATRIX_FORMAT_OWL);

  owlGroupBuildAccel(world);

  uint64_t topLevelBvhSizeInBytes = getAccelSizeInBytes(world);
  logSceneMemory(bottomLevelBvhSizeInBytes, topLevelBvhSizeInBytes, allocator);

  return world;

}

// For performance comaparisons, this version flattens each vox model into a single triangle mesh, 
OWLGroup Viewer::createFlatTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
    box3f &sceneBox )
{
  BufferAllocator allocator;

  // -------------------------------------------------------
  // declare triangle-based box geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "colorIndexPerBrick",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colorIndexPerBrick)},
    { "colorPalette",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colorPalette)},
    { "isFlat",  OWL_BOOL, OWL_OFFSETOF(TrianglesGeomData,isFlat)},
    { "primCountPerBrick",  OWL_INT, OWL_OFFSETOF(TrianglesGeomData,primCountPerBrick)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");


  assert(scene->num_instances > 0);
  assert(scene->num_models > 0);

  // Palette buffer is global to scene
  OWLBuffer paletteBuffer
    = allocator.deviceBufferCreate(context, OWL_UCHAR4, 256, scene->palette.color);

  // Cluster instances together that use the same model
  std::map<uint32_t, std::vector<uint32_t>> modelToInstances;

  for (uint32_t instanceIndex = 0; instanceIndex < scene->num_instances; instanceIndex++) {
    const ogt_vox_instance &vox_instance = scene->instances[instanceIndex];
    modelToInstances[vox_instance.model_index].push_back(instanceIndex);
  }

  std::vector<OWLGroup> geomGroups;
  geomGroups.reserve(scene->num_models);

  std::vector<owl::affine3f> instanceTransforms;
  instanceTransforms.reserve(scene->num_instances);

  size_t totalSolidVoxelCount = 0;
  size_t bottomLevelBvhSizeInBytes = 0;

  // Make instance transforms
  for (auto it : modelToInstances) {
    const ogt_vox_model *vox_model = scene->models[it.first];
    assert(vox_model);
    std::vector<uchar4> voxdata = extractSolidVoxelsFromModel(vox_model, enableCulling);

    LOG("building flat triangle geometry for model ...");

    std::vector<vec3f> meshVertices;
    meshVertices.reserve(voxdata.size() * NUM_BRICK_VERTICES);  // worst case
    std::vector<vec3i> meshIndices;
    meshIndices.reserve(voxdata.size() * NUM_BRICK_FACES);
    std::vector<unsigned char> colorIndicesPerBrick;
    colorIndicesPerBrick.reserve(voxdata.size());

    // Share vertices between voxels to save a little memory.  Only works for simple brick.
    constexpr bool SHARE_BRICK_VERTICES = (NUM_BRICK_VERTICES == 8 && NUM_BRICK_FACES == 12);
    std::map<std::tuple<int, int, int>, int> brickVertexToMeshVertexIndex;

    // Build mesh in object space where each brick is 1x1x1
    std::vector<int> indexRemap(NUM_BRICK_VERTICES);  // local brick vertex --> flat mesh vertex
    for (uchar4 voxel : voxdata) {
      const vec3i brickTranslation(voxel.x, voxel.y, voxel.z);
      for (int i = 0; i < NUM_BRICK_VERTICES; ++i) {
        const vec3f &v = brickVertices[i];
        std::tuple<int,int,int> brickVertex = std::make_tuple(
            brickTranslation.x + int(v.x),
            brickTranslation.y + int(v.y),
            brickTranslation.z + int(v.z));
        int vertexIndexInMesh = -1;
        if (SHARE_BRICK_VERTICES) {
          const auto it = brickVertexToMeshVertexIndex.find(brickVertex);
          if (it != brickVertexToMeshVertexIndex.end()) {
            vertexIndexInMesh = it->second;
          } else {
            meshVertices.push_back(vec3f(brickTranslation) + v);
            vertexIndexInMesh = int(meshVertices.size())-1;
            brickVertexToMeshVertexIndex[brickVertex] = vertexIndexInMesh;
          }
        } else {
          // do not share vertices
          meshVertices.push_back(vec3f(brickTranslation) + v);
          vertexIndexInMesh = int(meshVertices.size())-1;
        }
        indexRemap[i] = vertexIndexInMesh;  // brick vertex -> flat mesh vertex
      }
      for (const vec3i &index : brickIndices) {
        vec3i face(indexRemap[index.x], indexRemap[index.y], indexRemap[index.z]);
        meshIndices.push_back(face);
      }
      colorIndicesPerBrick.push_back(voxel.w);
    }

    LOG("Mesh vertex count: " << meshVertices.size());
    LOG("Mesh face count: " << meshIndices.size());

    OWLBuffer vertexBuffer
      = allocator.deviceBufferCreate(context, OWL_FLOAT3, meshVertices.size(), meshVertices.data());
    OWLBuffer indexBuffer
      = allocator.deviceBufferCreate(context, OWL_INT3, meshIndices.size(), meshIndices.data());

    OWLGeom trianglesGeom = owlGeomCreate(context, trianglesGeomType);
 
    owlTrianglesSetVertices(trianglesGeom, vertexBuffer, meshVertices.size(), sizeof(vec3f), 0);
    owlTrianglesSetIndices(trianglesGeom, indexBuffer, meshIndices.size(), sizeof(vec3i), 0);

    owlGeomSetBuffer(trianglesGeom, "vertex", vertexBuffer);
    owlGeomSetBuffer(trianglesGeom, "index", indexBuffer);
    owlGeomSetBuffer(trianglesGeom, "colorPalette", paletteBuffer);
    owlGeomSet1b(trianglesGeom, "isFlat", true);
    owlGeomSet1i(trianglesGeom, "primCountPerBrick", NUM_BRICK_FACES);

    OWLBuffer colorIndexBuffer
      = allocator.deviceBufferCreate(context, OWL_UCHAR, colorIndicesPerBrick.size(), colorIndicesPerBrick.data());
    owlGeomSetBuffer(trianglesGeom, "colorIndexPerBrick", colorIndexBuffer);

    // ------------------------------------------------------------------
    // the group/accel for the model
    // ------------------------------------------------------------------
    OWLGroup trianglesGroup = owlTrianglesGeomGroupCreate(context, 1, &trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    bottomLevelBvhSizeInBytes += getAccelSizeInBytes(trianglesGroup);

    LOG("adding (" << it.second.size() << ") instance transforms for model ...");
    for (uint32_t instanceIndex : it.second) {

      totalSolidVoxelCount += voxdata.size();

      const ogt_vox_instance &vox_instance = scene->instances[instanceIndex];

      const std::string instanceName = vox_instance.name ? vox_instance.name : "(unnamed)";
      if (vox_instance.hidden) {
        LOG("skipping hidden VOX instance: " << instanceName );
        continue;
      } else {
        LOG("building VOX instance: " << instanceName);
      }

      // VOX instance transform
      const ogt_vox_transform &vox_transform = vox_instance.transform;

      const owl::affine3f instanceTransform( 
          vec3f( vox_transform.m00, vox_transform.m01, vox_transform.m02 ),  // column 0
          vec3f( vox_transform.m10, vox_transform.m11, vox_transform.m12 ),  //  1
          vec3f( vox_transform.m20, vox_transform.m21, vox_transform.m22 ),  //  2
          vec3f( vox_transform.m30, vox_transform.m31, vox_transform.m32 )); //  3 

      // This matrix translates model to its center (in integer coords!) and applies instance transform
      const affine3f instanceMoveToCenterAndTransform = 
        affine3f(instanceTransform) * 
        affine3f::translate(-vec3f(float(vox_model->size_x/2),   // Note: snapping to int to match MV
                                   float(vox_model->size_y/2), 
                                   float(vox_model->size_z/2)));

      sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform, vec3f(0.0f)));
      sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform,  
            vec3f(float(vox_model->size_x), float(vox_model->size_y), float(vox_model->size_z))));

      instanceTransforms.push_back(instanceMoveToCenterAndTransform);

      geomGroups.push_back(trianglesGroup);

    }
  }

  LOG("Total solid voxels in all instanced models: " << totalSolidVoxelCount);

  const vec3f sceneCenter = sceneBox.center();
  const vec3f sceneSpan = sceneBox.span();

  if (this->enableGround) {
    // Extra scaled brick for ground plane
    
    OWLBuffer vertexBuffer
      = allocator.deviceBufferCreate(context,OWL_FLOAT3,NUM_BRICK_VERTICES,brickVertices);
    OWLBuffer indexBuffer
      = allocator.deviceBufferCreate(context,OWL_INT3,NUM_BRICK_FACES,brickIndices);

    OWLGeom trianglesGeom
      = owlGeomCreate(context,trianglesGeomType);
  
    owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
        NUM_BRICK_VERTICES,sizeof(vec3f),0);
    owlTrianglesSetIndices(trianglesGeom,indexBuffer,
        NUM_BRICK_FACES,sizeof(vec3i),0);

    owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
    owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
    owlGeomSetBuffer(trianglesGeom, "colorPalette", paletteBuffer);
    owlGeomSet1b(trianglesGeom, "isFlat", true);
    owlGeomSet1i(trianglesGeom, "primCountPerBrick", NUM_BRICK_FACES);

    std::vector<unsigned char> colorIndicesPerBrick = {249}; // grey in default palette
    OWLBuffer colorIndexBuffer
      = allocator.deviceBufferCreate(context, OWL_UCHAR, colorIndicesPerBrick.size(), colorIndicesPerBrick.data());
    owlGeomSetBuffer(trianglesGeom, "colorIndexPerBrick", colorIndexBuffer);

    OWLGroup trianglesGroup
      = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
    owlGroupBuildAccel(trianglesGroup);
    bottomLevelBvhSizeInBytes += getAccelSizeInBytes(trianglesGroup);

    geomGroups.push_back(trianglesGroup);

    instanceTransforms.push_back( 
        owl::affine3f::translate(vec3f(sceneCenter.x, sceneCenter.y, sceneCenter.z - 1 - 0.5f*sceneSpan.z)) *
        owl::affine3f::scale(vec3f(2*sceneSpan.x, 2*sceneSpan.y, 1.0f))*    // assume Z up
        owl::affine3f::translate(vec3f(-0.5f, -0.5f, 0.0f))
        );
  }

  // Apply final scene transform so we can use the same camera for every scene
  const float maxSpan = owl::reduce_max(sceneBox.span());
  owl::affine3f worldTransform = 
    owl::affine3f::scale(2.0f/maxSpan) *                                    // normalize
    owl::affine3f::translate(vec3f(-sceneCenter.x, -sceneCenter.y, -sceneBox.lower.z));  // center about (x,y) origin ,with Z up to match MV

  for (size_t i = 0; i < instanceTransforms.size(); ++i) {
    instanceTransforms[i] = worldTransform * instanceTransforms[i];
  }
  
  OWLGroup world = owlInstanceGroupCreate(context, 
      instanceTransforms.size(), 
      geomGroups.data(), 
      /*instanceIds*/ nullptr, 
      (const float*)instanceTransforms.data(), OWL_MATRIX_FORMAT_OWL);

  owlGroupBuildAccel(world);

  uint64_t topLevelBvhSizeInBytes = getAccelSizeInBytes(world);
  logSceneMemory(bottomLevelBvhSizeInBytes, topLevelBvhSizeInBytes, allocator);

  return world;

}

OWLGroup Viewer::createInstancedTriangleGeometryScene(OWLModule module, const ogt_vox_scene *scene, 
    owl::box3f &sceneBox)
{

  BufferAllocator allocator;

  // -------------------------------------------------------
  // declare triangle-based box geometry type
  // -------------------------------------------------------
  OWLVarDecl trianglesGeomVars[] = {
    { "index",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,index)},
    { "vertex", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,vertex)},
    { "colorIndexPerBrick",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colorIndexPerBrick)},
    { "colorPalette",  OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,colorPalette)},
    { /* sentinel to mark end of list */ }
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,-1);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

  LOG("building triangle geometry for single brick ...");


  // ------------------------------------------------------------------
  // triangle mesh for single unit brick
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = allocator.deviceBufferCreate(context,OWL_FLOAT3,NUM_BRICK_VERTICES,brickVertices);
  OWLBuffer indexBuffer
    = allocator.deviceBufferCreate(context,OWL_INT3,NUM_BRICK_FACES,brickIndices);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);
  
  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_BRICK_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_BRICK_FACES,sizeof(vec3i),0);
  
  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);

  OWLBuffer paletteBuffer
    = allocator.deviceBufferCreate(context, OWL_UCHAR4, 256, scene->palette.color);
  owlGeomSetBuffer(trianglesGeom, "colorPalette", paletteBuffer);
  
  // ------------------------------------------------------------------
  // the group/accel for a brick
  // ------------------------------------------------------------------
  OWLGroup trianglesGroup
    = owlTrianglesGeomGroupCreate(context,1,&trianglesGeom);
  owlGroupBuildAccel(trianglesGroup);
  uint64_t bottomLevelBvhSizeInBytes = getAccelSizeInBytes(trianglesGroup);


  // ------------------------------------------------------------------
  // instance the brick for each Vox model in the scene, flattening any Vox instances
  // ------------------------------------------------------------------

  LOG("building instances ...");

  std::vector<owl::affine3f> transformsPerBrick;
  std::vector<owl::affine3f> outlineTransformsPerBrick;
  std::vector<unsigned char> colorIndicesPerBrick;

  assert(scene->num_instances > 0);
  assert(scene->num_models > 0);

  size_t totalSolidVoxelCount = 0;

  for (uint32_t instanceIndex = 0; instanceIndex < scene->num_instances; instanceIndex++) {

    const ogt_vox_instance &vox_instance = scene->instances[instanceIndex];

    const std::string instanceName = vox_instance.name ? vox_instance.name : "(unnamed)";
    if (vox_instance.hidden) {
      LOG("skipping hidden VOX instance: " << instanceName );
      continue;
    } else {
      LOG("building VOX instance: " << instanceName);
    }

    const ogt_vox_model *vox_model = scene->models[vox_instance.model_index];
    assert(vox_model);

    // Note: for scenes with many instances of a model, cache this or rearrange loop
    std::vector<uchar4> voxdata = extractSolidVoxelsFromModel(vox_model, enableCulling);
    totalSolidVoxelCount += voxdata.size();

    // Color indices for this model
    for (size_t i = 0; i < voxdata.size(); ++i) {
        colorIndicesPerBrick.push_back(voxdata[i].w);
    }

    // VOX instance transform
    const ogt_vox_transform &vox_transform = vox_instance.transform;

    const owl::affine3f instanceTransform( 
        vec3f( vox_transform.m00, vox_transform.m01, vox_transform.m02 ),  // column 0
        vec3f( vox_transform.m10, vox_transform.m11, vox_transform.m12 ),  //  1
        vec3f( vox_transform.m20, vox_transform.m21, vox_transform.m22 ),  //  2
        vec3f( vox_transform.m30, vox_transform.m31, vox_transform.m32 )); //  3 

    // This matrix translates model to its center (in integer coords!) and applies instance transform
    const affine3f instanceMoveToCenterAndTransform = 
      affine3f(instanceTransform) * 
      affine3f::translate(-vec3f(float(vox_model->size_x/2),   // Note: snapping to int to match MV
                                 float(vox_model->size_y/2), 
                                 float(vox_model->size_z/2)));

    sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform, vec3f(0.0f)));
    sceneBox.extend(xfmPoint(instanceMoveToCenterAndTransform,  
          vec3f(float(vox_model->size_x), float(vox_model->size_y), float(vox_model->size_z))));

    // For outlines
    const affine3f instanceInflateOutline =
      affine3f::translate(vec3f(0.5f)) * affine3f::scale(vec3f(OUTLINE_SCALE)) * affine3f::translate(vec3f(-0.5f));

    for (size_t i = 0; i < voxdata.size(); ++i) {
      uchar4 b = voxdata[i];

      // Transform brick to its location in the scene
      owl::affine3f trans = instanceMoveToCenterAndTransform * owl::affine3f::translate(vec3f(b.x, b.y, b.z));
      transformsPerBrick.push_back(trans);
    }

    if (this->enableToonOutline) {
      for (size_t i = 0; i < voxdata.size(); ++i) {
        uchar4 b = voxdata[i];

        // insert outline
        owl::affine3f trans = instanceMoveToCenterAndTransform *
          owl::affine3f::translate(vec3f(b.x, b.y, b.z)) * instanceInflateOutline;
        outlineTransformsPerBrick.push_back(trans);
      }
    }

  }

  LOG("Total solid voxels in all instanced models: " << totalSolidVoxelCount);

  const vec3f sceneSpan = sceneBox.span();
  const vec3f sceneCenter = sceneBox.center();

  if (this->enableGround) {
    // Extra scaled brick for ground plane
    transformsPerBrick.push_back( 
        owl::affine3f::translate(vec3f(sceneCenter.x, sceneCenter.y, sceneCenter.z - 1 - 0.5f*sceneSpan.z)) *
        owl::affine3f::scale(vec3f(2*sceneSpan.x, 2*sceneSpan.y, 1.0f))*    // assume Z up
        owl::affine3f::translate(vec3f(-0.5f, -0.5f, 0.0f))
        );
    colorIndicesPerBrick.push_back(249);  // grey in default palette
  }

  // Apply final scene transform so we can use the same camera for every scene
  const float maxSpan = owl::reduce_max(sceneBox.span());
  owl::affine3f worldTransform = 
    owl::affine3f::scale(2.0f/maxSpan) *                                    // normalize
    owl::affine3f::translate(vec3f(-sceneCenter.x, -sceneCenter.y, -sceneBox.lower.z));  // center about (x,y) origin ,with Z up to match MV

  for (size_t i = 0; i < transformsPerBrick.size(); ++i) {
    transformsPerBrick[i] = worldTransform * transformsPerBrick[i];
  }
  for (size_t i = 0; i < outlineTransformsPerBrick.size(); ++i) {
    outlineTransformsPerBrick[i] = worldTransform * outlineTransformsPerBrick[i];
  }

  // Concat outline bricks onto regular bricks
  std::vector<uint8_t> visibilityMasks(transformsPerBrick.size() + outlineTransformsPerBrick.size());
  {
    size_t count = transformsPerBrick.size();
    transformsPerBrick.insert(transformsPerBrick.end(), outlineTransformsPerBrick.begin(), outlineTransformsPerBrick.end());
    std::fill(visibilityMasks.begin(), visibilityMasks.begin() + count, VISIBILITY_RADIANCE | VISIBILITY_SHADOW);
    std::fill(visibilityMasks.begin()+count, visibilityMasks.end(), VISIBILITY_OUTLINE);
  }

  // Set the color indices per brick now that the bricks have been fully instanced
  OWLBuffer colorIndexBuffer
      = allocator.deviceBufferCreate(context, OWL_UCHAR, colorIndicesPerBrick.size(), colorIndicesPerBrick.data());
  owlGeomSetBuffer(trianglesGeom, "colorIndexPerBrick", colorIndexBuffer);

  OWLGroup world = owlInstanceGroupCreate(context, transformsPerBrick.size(),
      nullptr, nullptr, (const float*)transformsPerBrick.data(), OWL_MATRIX_FORMAT_OWL);

  for (int i = 0; i < int(transformsPerBrick.size()); ++i) {
    owlInstanceGroupSetChild(world, i, trianglesGroup);  // All instances point to the same brick 
  }

  owlInstanceGroupSetVisibilityMasks(world, visibilityMasks.data());

  owlGroupBuildAccel(world);

  uint64_t topLevelBvhSizeInBytes = getAccelSizeInBytes(world);
  logSceneMemory(bottomLevelBvhSizeInBytes, topLevelBvhSizeInBytes, allocator);

  return world;
  
}

Viewer::Viewer(const ogt_vox_scene *scene, SceneType sceneType, const GlobalOptions &options)
  : enableGround(options.enableGround), 
  enableClipping(options.enableClipping), 
  enableToonOutline(options.enableToonOutline),
  enableCulling(options.enableCulling)
{
  // create a context on the first device:
  context = owlContextCreate(nullptr,1);
  OWLModule module = owlModuleCreate(context,ptxCode);

  // Set context options that affect program compilation
  // IMPORTANT: this needs to happen before any calls to owlBuildPrograms() !

  owlContextSetRayTypeCount(context, 3);  // primary, shadow, toon outline

  // Launch params
  OWLVarDecl launchVars[] = {

    { "enableToonOutline", OWL_BOOL, OWL_OFFSETOF(LaunchParams, enableToonOutline)},
    { "enableClipping", OWL_BOOL, OWL_OFFSETOF(LaunchParams, enableClipping)},

    { "frameID",       OWL_INT,    OWL_OFFSETOF(LaunchParams, frameID) }, 
    { "fbAccumBuffer", OWL_BUFPTR, OWL_OFFSETOF(LaunchParams, fbAccumBuffer) },
    { "fbPtr",         OWL_RAW_POINTER, OWL_OFFSETOF(LaunchParams, fbPtr) },
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(LaunchParams, fbSize)},

    { "world",         OWL_GROUP,  OWL_OFFSETOF(LaunchParams, world)},
    { "sunDirection",  OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, sunDirection)},
    { "sunColor",      OWL_FLOAT3, OWL_OFFSETOF(LaunchParams, sunColor)},
    { "brickScale",     OWL_FLOAT, OWL_OFFSETOF(LaunchParams, brickScale)},
    { "clipHeight",     OWL_INT, OWL_OFFSETOF(LaunchParams, clipHeight)},
    { /* sentinel to mark end of list */ }
  };

  // Tell OptiX which launch params will be constant across all frames with given values, so 
  // they can be specialized if possible during module compile.
  OWLBoundValueDecl boundValues[] = {
    { launchVars[0], &this->enableToonOutline},
    { launchVars[1], &this->enableClipping},
    { }
  };
  owlContextSetBoundLaunchParamValues(context, boundValues, -1);
  
  OWLGroup world;
  if (sceneType == SCENE_TYPE_FLAT) {
    world = createFlatTriangleGeometryScene(module, scene, this->sceneBox);
  } else if (sceneType == SCENE_TYPE_USER) {
    world = createUserGeometryScene(module, scene, this->sceneBox);
  } else {
    world = createInstancedTriangleGeometryScene(module, scene, this->sceneBox);
  }
  
  const vec3f sceneSpan = sceneBox.span();
  const int sceneLongestDim = indexOfMaxComponent(sceneSpan);
  const float brickScaleInWorldSpace = 2.f / sceneSpan[sceneLongestDim];
  this->clipHeight = int(sceneSpan.z) + 1;
  

  // ##################################################################
  // set miss and raygen program required for SBT
  // ##################################################################

  // -------------------------------------------------------
  // set up miss progs
  // -------------------------------------------------------
  OWLVarDecl missProgVars[]
    = {
    { "color0", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color0)},
    { "color1", OWL_FLOAT3, OWL_OFFSETOF(MissProgData,color1)},
    { /* sentinel to mark end of list */ }
  };
  // ----------- create object  ----------------------------
  OWLMissProg missProg = 
    owlMissProgCreate(context,module,"miss",sizeof(MissProgData), missProgVars,-1);
  owlMissProgSet(context, 0, missProg);
  
  // ----------- set variables  ----------------------------
  owlMissProgSet3f(missProg,"color0",owl3f{.8f,0.f,0.f});
  owlMissProgSet3f(missProg,"color1",owl3f{.8f,.8f,.8f});

  // Second program for shadow visibility
  OWLMissProg shadowMissProg = 
    owlMissProgCreate(context,module,"miss_shadow", 0u, nullptr,-1);
  owlMissProgSet(context, 1, shadowMissProg);

  // Can also use this for toon outline
  if (this->enableToonOutline) {
    owlMissProgSet(context, 2, shadowMissProg);
  }

  // -------------------------------------------------------
  // set up ray gen program
  // -------------------------------------------------------
  OWLVarDecl rayGenVars[] = {
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
  /* camera and frame buffer get set in resize() and cameraChanged() */
  
  launchParams = 
    owlParamsCreate(context, sizeof(LaunchParams), launchVars, -1);

  owlParamsSetGroup(launchParams, "world", world);
  owlParamsSet3f(launchParams, "sunColor", {1.f, 1.f, 1.f});
  owlParamsSet1f(launchParams, "brickScale", brickScaleInWorldSpace);
  owlParamsSet1i(launchParams, "clipHeight", clipHeight);

  // Also set runtime values for bound vars, just in case they can't be specialized
  owlParamsSet1b(launchParams, "enableToonOutline", this->enableToonOutline);
  owlParamsSet1b(launchParams, "enableClipping", enableClipping);

  // other params set at launch or resize.  Some of these may have also been bound
  // as constants earlier.
    
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
  if (sunDirty) {
    owl3f dir = { cos(sunPhi)*cos(sunTheta), sin(sunPhi)*cos(sunTheta), sin(sunTheta) };
    owlParamsSet3f(launchParams, "sunDirection", dir); 
    sunDirty = false;
    frameID = 0;
  }
  if (clipDirty && this->enableClipping) {
    owlParamsSet1i(launchParams, "clipHeight", this->clipHeight);
    clipDirty = false;
    frameID = 0;
  }
    
  // Since our progressive launches are faster than 60 fps on recent hardware, 
  // we loop over multiple subpixels here per display update.
  constexpr int NumSamples=4;
  for (int i = 0; i < NumSamples; ++i) {
    owlParamsSet1i(launchParams, "frameID", frameID);
    frameID++;
    owlLaunch2D(rayGen,fbSize.x,fbSize.y, launchParams);
  }
  owlLaunchSync(launchParams);
}

SceneType stringToSceneType(const char *s)
{
  std::string arg(s);
  if (arg == "flat") {
    return SCENE_TYPE_FLAT;
  } else if (arg == "user") {
    return SCENE_TYPE_USER;
  } else if (arg == "instanced") {
    return SCENE_TYPE_INSTANCED;
  }
  return SCENE_TYPE_INVALID;
}

bool stringToCamera(const char *s, vec3f &lookFrom, vec3f &lookAt, vec3f &lookUp)
{
  float v[9];
  if (sscanf_s(s, "%f %f %f  %f %f %f  %f %f %f", 
        v, v+1, v+2, v+3, v+4, v+5, v+6, v+7, v+8) == 9)
  {
    lookFrom = vec3f(v[0], v[1], v[2]);
    lookAt   = vec3f(v[3], v[4], v[5]);
    lookUp   = vec3f(v[6], v[7], v[8]);
    return true;
  }
  else if (sscanf_s(s, "%f %f %f  %f %f %f", 
        v, v+1, v+2, v+3, v+4, v+5 ) == 6)
  {
    lookFrom = vec3f(v[0], v[1], v[2]);
    lookAt   = vec3f(v[3], v[4], v[5]);
    return true;
  }
  else if (sscanf_s(s, "%f %f %f", 
        v, v+1, v+2) == 3)
  {
    lookFrom = vec3f(v[0], v[1], v[2]);
    return true;
  }
  return false;
}

int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  if (ac < 2) {
      LOG("need at least 1 expected argument for .vox file. Exiting");
      exit(1);
  }

  auto checkArgValue = [ac](int k, const std::string& arg) {
    if (k+1 >= ac) {
      LOG("Missing argument value for " << arg << ", exiting.");
      exit(1);
    }
  };

  Viewer::GlobalOptions options;
  SceneType sceneType = SCENE_TYPE_INSTANCED;

  vec3f lookFrom = init_lookFrom;
  vec3f lookAt = init_lookAt;
  vec3f lookUp = init_lookUp;

  std::vector<std::string> infiles;
  std::string outFileName;

  for (int i = 1; i < ac; ++i) {
    std::string arg = av[i];
    if (arg == "--no-ground") {
      options.enableGround = false;
    } 
    else if (arg == "--no-clipping") {
      // Might want to disable clipping in shader when measuring perf
      options.enableClipping = false;
    }
    else if (arg == "--no-outlines") {
      options.enableToonOutline = false;
    }
    else if (arg == "--cull-hidden") {
      options.enableCulling = true;
    }
    else if (arg == "--save") {
      checkArgValue(i, arg);
      outFileName = av[i+1];
      i++;
    }
    else if (arg == "--scenetype") {
      checkArgValue(i, arg);
      sceneType = stringToSceneType(av[i+1]);
      if (sceneType == SCENE_TYPE_INVALID) {
        LOG("Could not understand scene type: " << av[i+1] << ", exiting.");
        exit(1);
      }
      i++;  // skip arg value
    }
    else if (arg == "--camera") {
      checkArgValue(i, arg);
      if (!stringToCamera(av[i+1], lookFrom, lookAt, lookUp)) {
        LOG("Could not understand camera arguments, exiting.");
        exit(1);
      }
      i++;  // skip arg value
    }
    else if (arg[0] == '-') {
      LOG("Unrecognized argument: " << arg << ", exiting.");
      exit(1);
    }
    else {
      infiles.push_back(arg);  // assume all other args are file names
    }
  }
  // Expand possible directory
  if (infiles.size() == 1) {
    std::vector<std::string> directoryFiles;
    for (auto &p : std::filesystem::directory_iterator(infiles[0])) {
      directoryFiles.push_back(p.path().u8string());
    }
    if (directoryFiles.size() > 0) {
      infiles = std::move(directoryFiles);
    }
  }
  const ogt_vox_scene *scene = loadVoxScenes(infiles);

  if (!scene) {
      LOG("Could not read vox file: " << infiles[0].c_str() << ", exiting");
      exit(1);
  }
  if (scene->num_models == 0) {
      LOG("No voxel models found in file, exiting");
      ogt_vox_destroy_scene(scene);
      exit(1);
  }

  Viewer viewer(scene, sceneType, options);
  viewer.camera.setOrientation(lookFrom,
                               lookAt,
                               lookUp,
                               owl::viewer::toDegrees(acosf(init_cosFovy)));
  viewer.enableFlyMode();
  viewer.enableInspectMode(owl::box3f(vec3f(-1.f),vec3f(+1.f)));

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  if (!outFileName.empty()) {
    LOG("Rendering frames and exiting");
    viewer.showAndRun([&viewer, outFileName]() {
      if (viewer.frameID >= 4800) {
        viewer.screenShot(outFileName);
        return false;
      }
      return true; });
  } else {
    viewer.showAndRun();
  }

  ogt_vox_destroy_scene(scene);
}
