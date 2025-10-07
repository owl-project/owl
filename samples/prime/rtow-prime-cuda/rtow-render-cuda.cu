// ======================================================================== //
// Copyright 2019-2025 Ingo Wald                                            //
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

#include "World.h"
#include "FrameBuffer.h"

#include <random>
#include <cmath>
#include <random>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <memory>
#include <thread>

namespace samples {
  namespace device {
  
    struct PathState {
      vec3f  weight;
      Random random;
    };

    struct LaunchState {
      // one entry per pixel:
      Ray          *rays;
      Hit          *hits;
      device::PathState        *paths;

      // list of which rays are active
      int                      *activeRaysIn;
      int                      *activeRaysOut;
      // device-size *counter* of how many rays are active (serves as
      // atomic to write into d_activeRays)
      int                      *pNumActive;
    };
  
    inline __device__
    vec3f missColor(const vec3f &rayDir)
    {
      const float t = 0.5f*(rayDir.y + 1.0f);
      const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
      return c;
    }

    __global__
    void waveFrontPathCreate(Scene       world,
                             LaunchState launch,
                             int         sampleID)
    {
      const vec2i pixelID
        = vec2i(threadIdx)
        + vec2i(blockIdx)
        * vec2i(blockDim);
    
      if (pixelID.x >= world.fbSize.x) return;
      if (pixelID.y >= world.fbSize.y) return;
    
      const int tID = pixelID.x+world.fbSize.x*pixelID.y;
      
      PathState &path = launch.paths[tID];
      Ray       &ray  = launch.rays[tID];
      Hit       &hit  = launch.hits[tID];
        
      path.random.init(pixelID.x+sampleID*world.fbSize.x,
                       pixelID.y+sampleID*world.fbSize.y);
      path.weight = vec3f(1.f);
      
      const float u = float(pixelID.x + path.random()) / float(world.fbSize.x);
      const float v = float(pixelID.y + path.random()) / float(world.fbSize.y);
      ray = world.camera.generateRay(u,v,path.random);
      hit.primID = -1;
      hit.instID = -1;
    
      // and finally: initially every ray is active:
      launch.activeRaysOut[tID] = tID;
    }
    
    __global__
    void waveFrontAccumulate(Scene world,
                             LaunchState launch)
    {
      const vec2i pixelID
        = vec2i(threadIdx)
        + vec2i(blockIdx)
        * vec2i(blockDim);
    
      if (pixelID.x >= world.fbSize.x) return;
      if (pixelID.y >= world.fbSize.y) return;
    
      const int tID = pixelID.x+world.fbSize.x*pixelID.y;
      
      PathState &path        = launch.paths[tID];
        
      const vec3f addtl = (1.f/world.numSamplesPerPixel)*path.weight;
      vec3f &fb = world.fb[tID];
      atomicAdd(&fb.x,addtl.x);
      atomicAdd(&fb.y,addtl.y);
      atomicAdd(&fb.z,addtl.z);
    }
    

    __global__
    void waveFrontShade(int    numActive,
                        Scene  world,
                        LaunchState launch,
                        int    depth,
                        int    maxDepth)
    {
      const vec2i pixelID
        = vec2i(threadIdx)
        + vec2i(blockIdx)
        * vec2i(blockDim);
      const int linearThreadID = pixelID.x+world.fbSize.x*pixelID.y;
      if (linearThreadID >= numActive)
        return;
    
      const int rayID = launch.activeRaysIn[linearThreadID];
    
      PathState       &path   = launch.paths[rayID];
      Ray &ray    = launch.rays[rayID];
      Hit &hit    = launch.hits[rayID];

      if (hit.primID < 0) {
        if (depth == 0) 
          path.weight = missColor(normalize(ray.direction));
        return;
      }
    
      if (depth >= maxDepth) {
        path.weight = 0.f;
        return;
      }
      int instID = hit.instID;
      affine3f xfm = world.transforms[instID];
      auto &random = path.random;

      ScatterEvent scatter;
      vec3f &org = (vec3f&)ray.origin;
      vec3f &dir = (vec3f&)ray.direction;
      scatter.inDir = normalize(dir);
      int materialID = 0;
      // geomID==0 : triangles
      Triangle tri = world.triangles[hit.primID];
      tri.A = xfmPoint(xfm,tri.A);
      tri.B = xfmPoint(xfm,tri.B);
      tri.C = xfmPoint(xfm,tri.C);
      const float u = hit.u;
      const float v = hit.v;
#if HAVE_SHADING_NORMALS
      scatter.N
        = normalize((1.f-u-v) * tri.Na
                    + u       * tri.Nb
                    + v       * tri.Nc);
      if (dot(scatter.N,ray.direction) >= 0.f)
        scatter.N = normalize(cross(tri.B-tri.A,tri.C-tri.A));
#else
      scatter.N = normalize(cross(tri.B-tri.A,tri.C-tri.A));
#endif
      scatter.P =
        (1.f-u-v) * tri.A
        + u       * tri.B
        + v       * tri.C;
      materialID = world.materialIDs[instID];
      auto &material = world.materials[materialID];
      if (!material.scatter(scatter,random))
        // path lost .... ugh.
        return;
      
      org = scatter.out_org;
      dir = scatter.out_dir;
      ray.tmin = 1e-4f;
      ray.tmax = 1e20f;
      path.weight *= scatter.attenuation;
      int activeSlot = atomicAdd(launch.pNumActive,1);
      launch.activeRaysOut[activeSlot] = rayID;
    }
    
  } // ::device
  




  struct WaveFront {
    WaveFront(std::shared_ptr<World> world,
              const vec2i fbSize)
    {
      numAllocated = fbSize.x*fbSize.y;
      cudaMalloc(&d_launch.rays,numAllocated*sizeof(Ray));
      cudaMalloc(&d_launch.hits,numAllocated*sizeof(Hit));
      cudaMalloc(&d_launch.paths,numAllocated*sizeof(device::PathState));
      cudaMalloc(&d_launch.activeRaysIn,numAllocated*sizeof(int));
      cudaMalloc(&d_launch.activeRaysOut,numAllocated*sizeof(int));
      cudaMalloc(&d_launch.pNumActive,sizeof(int));
    }

    device::LaunchState d_launch;
    int numAllocated = 0;
  };

  void renderOneSample(const World &world,
                       const device::Scene &devScene,
                       WaveFront &waveFront,
                       // owl::prime::TraceContext *prime,
                       int sampleID, int numSamplesPerPixel)
  {
    cudaStream_t     stream = 0;//prime->getStream();
    vec2i tileSize = 16;

    device::LaunchState &launch = waveFront.d_launch;
    {
      vec2i numTiles = divRoundUp(devScene.fbSize,tileSize);
      device::waveFrontPathCreate<<<numTiles,tileSize,0,stream>>>
        (devScene,launch,sampleID);
    }

    const int numActiveAtStart = waveFront.numAllocated;
    int numActive = numActiveAtStart;
    for (int depth=0;numActive > 0;depth++) {
      
      // ------------------------------------------------------------------
      // trace active rays
      // ------------------------------------------------------------------
      std::swap(launch.activeRaysIn,launch.activeRaysOut);
      // owl::prime::RequestHandle query
      //   = prime->findClosestHit(launch.rays,
      //                           launch.hits,
      //                           launch.activeRaysIn,
      //                           numActive);
      opTraceIndexed(world.model,
                     numActive,
                     (int32_t*)launch.activeRaysIn,
                     numActiveAtStart,
                     (OPRay *)launch.rays,
                     (OPHit *)launch.hits,
                     0);
              
      
      // ------------------------------------------------------------------
      // shade active rays
      // ------------------------------------------------------------------
      // re-set numactive counter
      int zero = 0;
      cudaMemcpyAsync(launch.pNumActive,&zero,sizeof(zero),
                      cudaMemcpyHostToDevice,stream);
      vec2i numTiles = divRoundUp(devScene.fbSize,tileSize);
      device::waveFrontShade<<<numTiles,tileSize,0,stream>>>
        (numActive,devScene,launch,depth,devScene.maxPathLength); 
      // and read new active counter
      cudaMemcpyAsync(&numActive,launch.pNumActive,sizeof(zero),
                      cudaMemcpyDeviceToHost,stream);
      cudaStreamSynchronize(stream);
    }
    
    // ------------------------------------------------------------------
    // and finally, accumulate all:
    // ------------------------------------------------------------------
    {
      vec2i numTiles = divRoundUp(devScene.fbSize,tileSize);
      device::waveFrontAccumulate<<<numTiles,tileSize,0,stream>>>
        (devScene,launch);
    }
  }
    
  void renderFrame(std::shared_ptr<World> world,
                   std::shared_ptr<Camera> camera,
                   std::shared_ptr<FrameBuffer> fb,
                   int numSamplesPerPixel,
                   int maxPathLength// ,
                   // int numThreads
                   )
  {
    // ------------------------------------------------------------------
    // first, upload scene
    // ------------------------------------------------------------------

    device::Scene scene;
    scene.numSamplesPerPixel = numSamplesPerPixel;
    scene.maxPathLength = maxPathLength;

    // upload materials
    cudaMalloc(&scene.materials,world->materials.size()*sizeof(world->materials[0]));
    cudaMemcpy(scene.materials,world->materials.data(),
               world->materials.size()*sizeof(world->materials[0]),
               cudaMemcpyHostToDevice);

    // upload triangles
    cudaMalloc(&scene.triangles,world->triangles.size()*sizeof(world->triangles[0]));
    cudaMemcpy(scene.triangles,world->triangles.data(),
               world->triangles.size()*sizeof(world->triangles[0]),
               cudaMemcpyHostToDevice);

    // upload materialIDs
    cudaMalloc(&scene.materialIDs,world->materialIDs.size()*sizeof(world->materialIDs[0]));
    cudaMemcpy(scene.materialIDs,world->materialIDs.data(),
               world->materialIDs.size()*sizeof(world->materialIDs[0]),
               cudaMemcpyHostToDevice);

    // upload transforms
    cudaMalloc(&scene.transforms,world->transforms.size()*sizeof(world->transforms[0]));
    cudaMemcpy(scene.transforms,world->transforms.data(),
               world->transforms.size()*sizeof(world->transforms[0]),
               cudaMemcpyHostToDevice);
    
    scene.camera = *camera;
      
    // ------------------------------------------------------------------
    // create (and clear) device-side frame buffer
    // ------------------------------------------------------------------
    scene.fbSize = fb->size;
    cudaMalloc(&scene.fb,fb->size.x*fb->size.y*sizeof(vec3f));
    cudaMemset(scene.fb,0,fb->size.x*fb->size.y*sizeof(vec3f));

    WaveFront waveFront(world,fb->size);
    const vec2i fbSize = fb->size;
    for (int sampleID=0;
         sampleID<numSamplesPerPixel;
         sampleID+=1) 
      renderOneSample(*world,scene,waveFront,
                      sampleID,numSamplesPerPixel);
    
    cudaDeviceSynchronize();
    cudaMemcpy(fb->pixels.data(),scene.fb,
               fb->size.x*fb->size.y*sizeof(vec3f),
               cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
  }

  void renderFrame(std::shared_ptr<World> world,
                   std::shared_ptr<Camera> camera,
                   std::shared_ptr<FrameBuffer> fb,
                   int numSamplesPerPixel,
                   int maxPathLength);
  
  int rtow_CUDA(int ac, const char **av)
  {
    int Nx = 800, Ny = 600;
    int spp = 128;
    int maxPathLength = 3;
    
    for (int i=1;i<ac;i++) {
      const std::string arg = av[i];
      if (arg == "-fast" || arg == "-lq") {
        Nx = 800;
        Ny = 600;
        spp  = 16;
      } else if (arg == "--final" || arg == "-hq") {
        Nx = 2*800;
        Ny = 2*600;
        spp  = 1024;
      } else if (arg == "--size") {
        Nx = std::stoi(av[++i]);
        Ny = std::stoi(av[++i]);
      } else if (arg == "--rec" || arg == "--bounces" || arg == "--max-path-length") {
        maxPathLength = std::stoi(av[++i]);;
      } else if (arg == "--spp" || arg == "-spp") {
        spp = std::stoi(av[++i]);;
      } else throw std::runtime_error("unknown arg " +arg);
    }
    
    // create - and set - the camera
    const vec3f lookfrom(13.f, 2.f, 3.f);
    const vec3f lookat(0.f, 0.f, 0.f);
    std::shared_ptr<Camera>
      camera = std::make_shared<Camera>(lookfrom,
                                        lookat,
                                        /* up */ vec3f(0.f, 1.f, 0.f),
                                        /* fovy, in degrees */ 20.f,
                                        /* aspect */ float(Nx) / float(Ny),
                                        /* aperture */ 0.1f,
                                        /* dist to focus: */ 10.f);
    
    // create a frame buffer
    std::shared_ptr<FrameBuffer> fb
      = std::make_shared<FrameBuffer>(vec2i(Nx, Ny));
    Random random;
    std::shared_ptr<World> world
      = createScene(random);
    world->finalize();
    
    // render the frame (and time it)
    auto t0 = std::chrono::system_clock::now();
    renderFrame(world,camera,fb,
                spp,maxPathLength// ,
                // numThreads
                );
    auto t1 = std::chrono::system_clock::now();
    std::cout << "done rendering, which took "
              << std::setprecision(4) << std::chrono::duration<double>(t1-t0).count()
              << " seconds (for " << spp
              << " paths per pixel)" << std::endl;
       
    savePNG("finalChapter.png",*fb);

    // ... done.
    return 0;
  }

}
  

int main(int ac, const char **av)
{ return samples::rtow_CUDA(ac,av); }
