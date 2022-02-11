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

// This program sets up a single geometric object, a mesh for a cube, and
// its acceleration structure, then ray traces it.

// public owl node-graph API
#include "owl/owl.h"
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
#include <random>


#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char deviceCode_ptx[];

const int NUM_VERTICES = 24;
// Making vertices redundant to apply different texcoords depending
// on the face
vec3f vertices[NUM_VERTICES] =
  {
    // back face 
    { -1.f,-1.f,-1.f }, // bottom left far    0
    { +1.f,-1.f,-1.f }, // bottom right far   1
    { -1.f,+1.f,-1.f }, // top left far       2
    { +1.f,+1.f,-1.f }, // top right far      3
    // front face
    { -1.f,-1.f,+1.f }, // bottom left near   4
    { +1.f,-1.f,+1.f }, // bottom right near  5
    { -1.f,+1.f,+1.f }, // top left near      6
    { +1.f,+1.f,+1.f }, // top right near     7
    // top face
    { -1.f,+1.f,-1.f }, // top left far       8
    { +1.f,+1.f,-1.f }, // top right far      9
    { -1.f,+1.f,+1.f }, // top left near      10
    { +1.f,+1.f,+1.f }, // top right near     11
    // bottom face
    { -1.f,-1.f,-1.f }, // bottom left far    12
    { +1.f,-1.f,-1.f }, // bottom right far   13
    { -1.f,-1.f,+1.f }, // bottom left near   14
    { +1.f,-1.f,+1.f }, // bottom right near  15
    // left face
    { -1.f,-1.f,-1.f }, // bottom left far    16
    { -1.f,+1.f,-1.f }, // top left far       17
    { -1.f,-1.f,+1.f }, // bottom left near   18
    { -1.f,+1.f,+1.f }, // top left near      19
    // right face
    { +1.f,-1.f,-1.f }, // bottom right far   20
    { +1.f,+1.f,-1.f }, // top right far      21
    { +1.f,-1.f,+1.f }, // bottom right near  22
    { +1.f,+1.f,+1.f }  // top right near     23
  };

vec2f texCoords[NUM_VERTICES] =
  {
    // back
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f },

    // front
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f },

    // top
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f },

    // bottom
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f },

    // left
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f },

    // right
    { +0.f,+0.f },
    { +0.f,+1.f },
    { +1.f,+0.f },
    { +1.f,+1.f }
  };

const int NUM_INDICES = 12;
vec3i indices[NUM_INDICES] =
  {
    // back face
    { 0,1,3 }, { 2,3,0 },
    // front face
    { 5,7,6 }, { 5,6,4 },
    // bottom face
    { 12,14,15 }, { 12,15,13 },
    // top face
    { 8,9,11 }, { 8,11,10 },
    // left face
    { 18,16,17 }, { 18,17,19 },
    // right face
    { 20,22,23 }, { 20,23,21 }
  };

const char *outFileName = "s11-textures.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
const float cosFovy = 0.66f;

std::default_random_engine rndGen;
std::uniform_real_distribution<float> distribution_uniform(0.0,1.0);
std::uniform_real_distribution<float> distribution_speed(.1f,.8f);
std::uniform_int_distribution<int> distribution_texSize(2,16);

int main(int ac, char **av)
{
  LOG("owl::ng example '" << av[0] << "' starting up");

  // create a context on the first device:
  OWLContext context = owlContextCreate(nullptr,1);
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
    { "texCoord", OWL_BUFPTR, OWL_OFFSETOF(TrianglesGeomData,texCoord)},
    { "texture",  OWL_TEXTURE, OWL_OFFSETOF(TrianglesGeomData,texture)}
  };
  OWLGeomType trianglesGeomType
    = owlGeomTypeCreate(context,
                        OWL_TRIANGLES,
                        sizeof(TrianglesGeomData),
                        trianglesGeomVars,4);
  owlGeomTypeSetClosestHit(trianglesGeomType,0,
                           module,"TriangleMesh");

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // triangle mesh
  // ------------------------------------------------------------------
  OWLBuffer vertexBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT3,NUM_VERTICES,vertices);
  OWLBuffer texCoordsBuffer
    = owlDeviceBufferCreate(context,OWL_FLOAT2,NUM_VERTICES,texCoords);
  OWLBuffer indexBuffer
    = owlDeviceBufferCreate(context,OWL_INT3,NUM_INDICES,indices);
  OWLBuffer frameBuffer
    = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);

  OWLGeom trianglesGeom
    = owlGeomCreate(context,trianglesGeomType);

  owlTrianglesSetVertices(trianglesGeom,vertexBuffer,
                          NUM_VERTICES,sizeof(vec3f),0);
  owlTrianglesSetIndices(trianglesGeom,indexBuffer,
                         NUM_INDICES,sizeof(vec3i),0);

  owlGeomSetBuffer(trianglesGeom,"vertex",vertexBuffer);
  owlGeomSetBuffer(trianglesGeom,"index",indexBuffer);
  owlGeomSetBuffer(trianglesGeom,"texCoord",texCoordsBuffer);

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
  OWLGroup world
    = owlInstanceGroupCreate(context,1,&trianglesGroup);
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
    { "fbPtr",         OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)},
    { "fbSize",        OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize)},
    { "world",         OWL_GROUP,  OWL_OFFSETOF(RayGenData,world)},
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
    { /* sentinel to mark end of list */ }
  };

  // ----------- create object  ----------------------------
  OWLRayGen rayGen
    = owlRayGenCreate(context,module,"simpleRayGen",
                      sizeof(RayGenData),
                      rayGenVars,-1);

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
  owlRayGenSetBuffer(rayGen,"fbPtr",        frameBuffer);
  owlRayGenSet2i    (rayGen,"fbSize",       (const owl2i&)fbSize);
  owlRayGenSetGroup (rayGen,"world",        world);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  owlBuildPrograms(context);
  owlBuildPipeline(context);
  owlBuildSBT(context);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################

  LOG("launching ...");
  owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);

  LOG("done with launch, writing picture ...");
  // for host pinned mem it doesn't matter which device we query...
  const uint32_t *fb
    = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  assert(fb);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);
  // ##################################################################
  // and finally, clean up
  // ##################################################################

  LOG("destroying devicegroup ...");
  owlContextDestroy(context);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
