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

// This program shows a minimal setup: no geometry, just a ray generation
// shader that accesses the pixels and draws a checkerboard pattern to
// the output file ll00-rayGenOnly.png

// public owl API
#include <owl/owl.h>
// our device-side data structures
#include "deviceCode.h"
#include <vector>
// external helper stuff for image output
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

#define LOG(message)                                            \
  std::cout << OWL_TERMINAL_BLUE;                               \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;
#define LOG_OK(message)                                         \
  std::cout << OWL_TERMINAL_LIGHT_BLUE;                         \
  std::cout << "#owl.sample(main): " << message << std::endl;   \
  std::cout << OWL_TERMINAL_DEFAULT;

extern "C" char ptxCode[];

// When run, this program produces this PNG as output.
// In this case the correct result is a red and light gray checkerboard,
// as nothing is actually rendered
const char *outFileName = "s09-dvr.png";
// image resolution
const vec2i fbSize(800,600);
const vec3f lookFrom(-2.f,-1.5f,-1.f);
const vec3f lookAt(.5f,.5f,.5f);
const vec3f lookUp(0.f,0.f,1.f);
const float cosFovy = 0.36f;

//const vec3i     dims = vec3i(1024);
const vec3i     dims = vec3i(2500);
//const vec3i     dims = vec3i(2048);

void loadVolume(float *voxels,
                const vec3i &dims)
{
  std::vector<float> magnetic(512*512*512);
  FILE *file = fopen("magnetic.raw","rb");
  if (!file) throw std::runtime_error("could not open volume file");
  fread(magnetic.data(),magnetic.size(),sizeof(float),file);
  for (int iz=0;iz<dims.z;iz++)
    for (int iy=0;iy<dims.y;iy++)
      for (int ix=0;ix<dims.x;ix++) {
        vec3f rel = vec3f(ix+.5f,iy+.5f,iz+.5f) * (1.f/vec3f(dims.x,dims.y,dims.z));
        vec3f mag = 511.f * rel;
        int ix0 = std::min(int(mag.x)+0,511);
        int iy0 = std::min(int(mag.y)+0,511);
        int iz0 = std::min(int(mag.z)+0,511);
        int ix1 = std::min(ix0+1,511);
        int iy1 = std::min(iy0+1,511);
        int iz1 = std::min(iz0+1,511);
        const float v000 = magnetic[ix0 + 512*iy0 + 512*512*iz0];
        const float v001 = magnetic[ix1 + 512*iy0 + 512*512*iz0];
        const float v010 = magnetic[ix0 + 512*iy1 + 512*512*iz0];
        const float v011 = magnetic[ix1 + 512*iy1 + 512*512*iz0];
        const float v100 = magnetic[ix0 + 512*iy0 + 512*512*iz1];
        const float v101 = magnetic[ix1 + 512*iy0 + 512*512*iz1];
        const float v110 = magnetic[ix0 + 512*iy1 + 512*512*iz1];
        const float v111 = magnetic[ix1 + 512*iy1 + 512*512*iz1];
        const float fx = mag.x - trunc(mag.x);
        const float fy = mag.y - trunc(mag.y);
        const float fz = mag.z - trunc(mag.z);

        const float v00x = (1.f-fx) * v000 + fx * v001;
        const float v01x = (1.f-fx) * v010 + fx * v011;
        const float v10x = (1.f-fx) * v100 + fx * v101;
        const float v11x = (1.f-fx) * v110 + fx * v111;

        const float v0yx = (1.f-fy) * v00x + fy * v01x;
        const float v1yx = (1.f-fy) * v10x + fy * v11x;

        const float vzyx = (1.f-fz) * v0yx + fz * v1yx;
        
        const size_t idx = ix+size_t(dims.x)*(iy+size_t(dims.y)*(iz));
        voxels[idx] = vzyx;//cosf(4.f*rel.x) * cosf(3.f*rel.y+2.f*rel.z) ;
      }
}

int main(int ac, char **av)
{
  // The output window will show comments for many of the methods called.
  // Walking through the code line by line with a debugger is educational.
  LOG("owl example '" << av[0] << "' starting up");

  // ##################################################################
  // set up all the *CODE* we want to run
  // ##################################################################

  LOG("building module, programs, and pipeline");

  // Initialize CUDA and OptiX 7, and create an "owl device," a context to hold the
  // ray generation shader and output buffer. The "1" is the number of devices requested.
  OWLContext owl
    = owlContextCreate(nullptr,0);
  // PTX is the intermediate code that the CUDA deviceCode.cu shader program is converted into.
  // You can see the machine-centric PTX code in
  // build\samples\s00-rayGenOnly\cuda_compile_ptx_1_generated_deviceCode.cu.ptx_embedded.c
  // This PTX intermediate code representation is then compiled into an OptiX module.
  // See https://devblogs.nvidia.com/how-to-get-started-with-optix-7/ for more information.
  OWLModule module
    = owlModuleCreate(owl,ptxCode);


  size_t    numVoxels = size_t(dims.x)*size_t(dims.y)*size_t(dims.z);
  OWLBuffer voxelsBuffer = owlManagedMemoryBufferCreate(owl,OWL_FLOAT,numVoxels,nullptr);
  float    *voxels = (float*)owlBufferGetPointer(voxelsBuffer,0);
  loadVolume(voxels,dims);
  
  
  OWLVarDecl rayGenVars[]
    = {
       { "fbPtr",  OWL_BUFPTR, OWL_OFFSETOF(RayGenData,fbPtr)  },
       { "fbSize", OWL_INT2,   OWL_OFFSETOF(RayGenData,fbSize) },
    { "camera.pos",    OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.pos)},
    { "camera.dir_00", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_00)},
    { "camera.dir_du", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_du)},
    { "camera.dir_dv", OWL_FLOAT3, OWL_OFFSETOF(RayGenData,camera.dir_dv)},
       { "dims",   OWL_INT3,   OWL_OFFSETOF(RayGenData,dims)   },
       { "voxels", OWL_BUFPTR, OWL_OFFSETOF(RayGenData,voxels) },
       { /* sentinel: */ nullptr }
  };
  // Allocate room for one RayGen shader, create it, and
  // hold on to it with the "owl" context
  OWLRayGen rayGen
    = owlRayGenCreate(owl,module,"renderDVR",
                      sizeof(RayGenData),rayGenVars,-1);
 
  // (re-)builds all optix programs, with current pipeline settings 
  owlBuildPrograms(owl);
  // Create the pipeline. Note that owl will (kindly) warn there are no geometry and no miss programs defined.
  owlBuildPipeline(owl);

  // ------------------------------------------------------------------
  // alloc buffers
  // ------------------------------------------------------------------
  LOG("allocating frame buffer");
  // Create a frame buffer as page-locked, aka "pinned" memory. See CUDA documentation for benefits and more info.
  OWLBuffer
  frameBuffer = owlHostPinnedBufferCreate(owl,
                                          /*type:*/OWL_INT,
                                          /*size:*/fbSize.x*fbSize.y);

  // ------------------------------------------------------------------
  // build Shader Binding Table (SBT) required to trace the groups
  // ------------------------------------------------------------------

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


  owlRayGenSetBuffer(rayGen,"fbPtr",frameBuffer);
  owlRayGenSet2i(rayGen,"fbSize",fbSize.x,fbSize.y);
  owlRayGenSet3i(rayGen,"dims",dims.x,dims.y,dims.z);
  owlRayGenSetBuffer(rayGen,"voxels",voxelsBuffer);
  owlRayGenSet3f    (rayGen,"camera.pos",   (const owl3f&)camera_pos);
  owlRayGenSet3f    (rayGen,"camera.dir_00",(const owl3f&)camera_d00);
  owlRayGenSet3f    (rayGen,"camera.dir_du",(const owl3f&)camera_ddu);
  owlRayGenSet3f    (rayGen,"camera.dir_dv",(const owl3f&)camera_ddv);

  // Build a shader binding table entry for the ray generation record.
  owlBuildSBT(owl);

  // ##################################################################
  // now that everything is ready: launch it ....
  // ##################################################################
  
  LOG("executing the launch ...");
  // Normally launching without a hit or miss shader causes OptiX to trigger warnings.
  // Owl's wrapper call here will set up fake hit and miss records into the SBT to avoid these.

  double t0 = getCurrentTime();
  int numTimesRendered = 0;
  while (1) {
    owlRayGenLaunch2D(rayGen,fbSize.x,fbSize.y);
    numTimesRendered++;
    double t1 = getCurrentTime();
    if (t1 - t0 > 100.f) {
      std::cout << "rendered " << numTimesRendered << " frames in " << (t1-t0) << " secs" << std::endl;
      break;
    }
  }
 
  LOG("done with launch, writing frame buffer to " << outFileName);
  const uint32_t *fb = (const uint32_t*)owlBufferGetPointer(frameBuffer,0);
  stbi_write_png(outFileName,fbSize.x,fbSize.y,4,
                 fb,fbSize.x*sizeof(uint32_t));
  LOG_OK("written rendered frame buffer to file "<<outFileName);

  // ##################################################################
  // and finally, clean up
  // ##################################################################
  
  LOG("cleaning up ...");
  owlModuleRelease(module);
  owlRayGenRelease(rayGen);
  owlBufferRelease(frameBuffer);
  owlContextDestroy(owl);
  
  LOG_OK("seems all went OK; app is done, this should be the last output ...");
}
