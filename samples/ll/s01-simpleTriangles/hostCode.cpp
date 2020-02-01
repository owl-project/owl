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

// ll01-simpleTriangles: ray trace a cube made of triangles

// public owl-ll API
#include <owl/ll.h>
// our device-side data structures
#include "deviceCode.h"
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                    \
  std::cout << OWL_TERMINAL_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                    \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                       \
  std::cout << "#ll.sample(main): " << message << std::endl;  \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

// data for a mesh forming a cube
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

const char *outFileName = "ll01-simpleTriangles.png";
const vec2i fbSize(800,600);
const vec3f lookFrom(-4.f,-3.f,-2.f);
const vec3f lookAt(0.f,0.f,0.f);
const vec3f lookUp(0.f,1.f,0.f);
// cosine of field of view from top to bottom of the image plane,
// e.g., cos(90 degrees) = 1.0f
const float cosFovy = 0.66f;    // about 48.7 degrees

int main(int ac, char **av)
{
  LOG("ll example '" << av[0] << "' starting up");

  // for what these initial methods do, see comments in s00-rayGenOnly\hostCode.cpp;
  // comments are added here only for code that differs from that code.
  LLOContext llo = lloContextCreate(nullptr,0);

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");
  
  lloAllocModules(llo,1);
  lloModuleCreate(llo,0,ptxCode);
  lloBuildModules(llo);

  // Set up a closest-hit shader for triangles
  enum { TRIANGLES_GEOM_TYPE=0,NUM_GEOM_TYPES };
  // Allocate memory for the number of different types of geometry to be
  // intersected by rays. In this case, just one: triangle meshes.
  // TODO: move the following to a later sample.
  // Each Geometry Type has one or more types of rays associated with it: eye rays,
  // shadow rays, etc. For each of these ray types, there is a "hit group", aka
  // "program group", consisting of a closest hit shader (and possibly an any-hit
  // shader and a custom intersection shader) that is set for it.
  // So, an eye ray can (and likely should) have a different closest hit shader
  // than a shadow ray. In this sample we have only eye rays. The any-hit shader
  // will come into play for semitransparent objects, and the intersection shader
  // be used for objects that are not triangle meshes. Triangle meshes have a built-in
  // (and usually hardware-accelerated) intersection shader.
  lloAllocGeomTypes(llo,NUM_GEOM_TYPES);
  // In OWL's llo context, store the size of the data structure associated
  // with the triangle mesh. See deviceCode.h for what is in this structure;
  // in this case it is a color for the mesh, and pointers to the index and
  // vertex buffers.
  lloGeomTypeCreate(llo,TRIANGLES_GEOM_TYPE,sizeof(TrianglesGeomData));
  // Set the hit group's closest hit shader for the eye rays we'll be shooting.
  // When a ray hits the triangle mesh, this shader will be executed.
  lloGeomTypeClosestHit(llo,
                        /*program ID*/TRIANGLES_GEOM_TYPE,
                        /*ray type  */0,    // we are using only one ray type, from the eye
                        /*module:*/0,
                        "TriangleMesh");
  

  // Allocate room for and create one ray generation shader for the scene.
  // TODO: move the following to a later sample.
  // Note that this shader and the miss shader do not have a ray type, as they
  // are associated only with a module, not a geometry type. 
  lloAllocRayGens(llo,1);
  lloRayGenCreate(llo,
                  /*program ID*/0,
                  /*module:*/0,
                  "simpleRayGen",
                  sizeof(RayGenData));
  
  // Allocate room for and create a miss shader for the scene.
  // If no triangle is hit by a ray from the eye, this shader is invoked.
  lloAllocMissProgs(llo,1);
  lloMissProgCreate(llo,
                    /*program ID*/0,
                    /*module:*/0,
                    "miss",
                    sizeof(MissProgData));

  lloBuildPrograms(llo);
  lloCreatePipeline(llo);

  // ##################################################################
  // set up all the *GEOMS* we want to run that code on
  // ##################################################################

  LOG("building geometries ...");

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  enum { VERTEX_BUFFER=0,INDEX_BUFFER,FRAME_BUFFER,NUM_BUFFERS };
  // allocate buffers for the triangle mesh's vertex and index buffer,
  // along with the frame buffer
  lloAllocBuffers(llo,NUM_BUFFERS);
  // we do not need to make pinned memory for the triangle buffers, as we
  // will not be reading back from them. TODO - true?
  lloDeviceBufferCreate(llo,VERTEX_BUFFER,NUM_VERTICES*sizeof(vec3f),vertices);
  lloDeviceBufferCreate(llo,INDEX_BUFFER,NUM_INDICES*sizeof(vec3i),indices);
  lloHostPinnedBufferCreate(llo,FRAME_BUFFER,fbSize.x*fbSize.y*sizeof(uint32_t));
  
  // ------------------------------------------------------------------
  // alloc geom
  // ------------------------------------------------------------------
  enum { TRIANGLES_GEOM=0,NUM_GEOMS };
  // Resize the array of geometry IDs; in this case, just one is needed, for our single mesh.
  lloAllocGeoms(llo,NUM_GEOMS);
  // Allocate room for geometry ID==0 to store a triangle mesh in a TrianglesGeom structure.
  lloTrianglesGeomCreate(llo,
                         /* geom ID    */TRIANGLES_GEOM,
                         /* type/PG ID */TRIANGLES_GEOM_TYPE);
  // Fill in this ID==0 geometry location with the vertex and index buffer defining the mesh.
  lloTrianglesGeomSetVertexBuffer(llo,
                                  /* geom ID   */TRIANGLES_GEOM,
                                  /* buffer ID */VERTEX_BUFFER,
                                  /* meta info */NUM_VERTICES,sizeof(vec3f),0);
  lloTrianglesGeomSetIndexBuffer(llo,
                                 /* geom ID   */TRIANGLES_GEOM,
                                 /* buffer ID */INDEX_BUFFER,
                                 /* meta info */NUM_INDICES,sizeof(vec3i),0);

  // ##################################################################
  // set up all *ACCELS* we need to trace into those groups
  // ##################################################################

  enum { TRIANGLES_GROUP=0,NUM_GROUPS };
  // We have just one triangle mesh, so note its ID in a single-entry array
  lloAllocGroups(llo,NUM_GROUPS);
  int geomsInGroup[] = { 0 };
  lloTrianglesGeomGroupCreate(llo,
                              /* group ID */TRIANGLES_GROUP,
                              /* geoms in group, pointer */ geomsInGroup,
                              /* geoms in group, count   */ 1);
  // build an acceleration structure for this group of triangles
  // TODO - what's this acceleration structure, exactly? Is it a BVH around the
  // triangles in the mesh, or a BVH around the single mesh itself (i.e., a box around the whole mesh)?
  // I guess I'll see - this example is so simple it's hard to tell!
  lloGroupAccelBuild(llo,TRIANGLES_GROUP);

  // ##################################################################
  // build *SBT* required to trace the groups
  // ##################################################################
  LOG("building SBT ...");

  // ----------- build hitgroups -----------
  // the triangle mesh makes its data available this way
  lloSbtHitProgsBuild
    (llo,
     [&](uint8_t *output,int devID,int geomID,int rayID) {
      TrianglesGeomData &self = *(TrianglesGeomData*)output;
      self.color  = vec3f(0,1,0);
      self.index  = (vec3i*)lloBufferGetPointer(llo,INDEX_BUFFER,devID);
      self.vertex = (vec3f*)lloBufferGetPointer(llo,VERTEX_BUFFER,devID);
    });
  
  // ----------- build miss prog(s) -----------
  // TODO I'm a bit lost here: the comment "we don't have any" is interesting,
  // and I thought miss programs were per module (as allocated further up), not per ray type
  lloSbtMissProgsBuild
    (llo,
     [&](uint8_t *output,
         int devID,
         int rayType) {
      /* we don't have any ... */
      ((MissProgData*)output)->color0 = vec3f(.8f,0.f,0.f);
      ((MissProgData*)output)->color1 = vec3f(.8f,.8f,.8f);
    });
  
  // ----------- build raygens -----------
  // initialize the camera with vectors for determining the ray's direction,
  // given an input pixel location
  lloSbtRayGensBuild
    (llo,[&](uint8_t *output,
             int devID,
             int rgID) {
      RayGenData *rg = (RayGenData*)output;
      rg->deviceIndex   = devID;
      rg->deviceCount = lloGetDeviceCount(llo);
      rg->fbSize = fbSize;
      rg->fbPtr  = (uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,devID);
      rg->world  = lloGroupGetTraversable(llo,TRIANGLES_GROUP,devID);

      // Compute camera frame.
      // dir_00 is the upper left corner of the image, with dir_du being the
      // 3D change per pixel going to the right, dir_dv the change going down.
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
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  lloLaunch2D(llo,0,fbSize.x,fbSize.y);

  LOG("done with launch, writing picture ...");
  // for host pinned memory it doesn't matter which device we query,
  // since all GPUs will share the same storage area.
  const uint32_t *fb = (const uint32_t*)lloBufferGetPointer(llo,FRAME_BUFFER,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("destroying llo context ...");
  lloContextDestroy(llo);

  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
