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

#include "deviceCode.h"
#include <optix_device.h>
#include <owl/common/math/random.h>

typedef owl::common::LCG<4> Random;

__constant__ Globals optixLaunchParams;

struct Hit {
  bool  hadHit = false;
  vec3f pos, nor, col;
};

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  uint64_t clock_begin = clock();

  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::RayT</*ray type*/0,/*total ray types*/1, /*disable geometry contribution*/ true> ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  Hit hit;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/hit);

  uint64_t clock_end = clock();
  // if (optixLaunchParams.heatmapEnabled) 
  // float heatmapScale = 10000;
  // {
  //   float t = (clock_end-clock_begin)*(1.f / heatmapScale);
  //   hit.col = make_float4(t, t, t, 1.f);
  // }

  
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(hit.col);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  Hit &prd = owl::getPRD<Hit>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  const vec3f rayOrg = optixGetWorldRayOrigin();
  const vec3f rayDir = normalize((vec3f)optixGetWorldRayDirection());

  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  vec3f Ng     = normalize(cross(B-A,C-A));
  Ng = optixTransformNormalFromObjectToWorldSpace(Ng);
  if (dot(Ng,rayDir) > 0.f) Ng = -Ng;

  const vec2f uv     = optixGetTriangleBarycentrics();
  const vec2f tc
    = (1.f-uv.x-uv.y)*self.texCoord[index.x]
    +      uv.x      *self.texCoord[index.y]
    +           uv.y *self.texCoord[index.z];
  vec4f texColor = tex2D<float4>(self.texture,tc.x,tc.y);

  const vec3f P = rayOrg + (optixGetRayTmax()*.999f) * rayDir;

  const vec3f lightDir(1,1,1);
  bool illuminated = false;
  if (dot(lightDir,Ng) > 0.f) {
    illuminated = true;
  }

  float weight = .1f;
  weight += .2f*fabs(dot(rayDir,Ng));
  if (illuminated)
    weight += 1.5f * dot(normalize(lightDir),Ng);
  
  prd.col = weight * vec3f(texColor);
  prd.hadHit = true;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  Hit &prd = owl::getPRD<Hit>();
  const float t = pixelID.y / (float)optixGetLaunchDimensions().y;
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  prd.hadHit = false;
  prd.col = c;
}

OPTIX_INSTANCE_PROGRAM(instanceProg)(const int32_t instanceIndex, OptixInstance &i)
{
  float t = optixLaunchParams.time;
  Random rng(vec2i(instanceIndex / 1000, instanceIndex % 1000));
  
  // select a BLAS at random
  uint32_t blasID = int(instanceIndex + int(t)) % optixLaunchParams.numBLAS;
  // if (instanceIndex == 0) printf("blasID = %i, time %f\n", optixLaunchParams.numBLAS, optixLaunchParams.time);

  if (instanceIndex == 0) printf("num boxes %d %d %d\n", optixLaunchParams.numBoxes.x, optixLaunchParams.numBoxes.y, optixLaunchParams.numBoxes.z); 

  i.sbtOffset = optixLaunchParams.BLASOffsets[blasID];
  i.traversableHandle = optixLaunchParams.BLAS[blasID];

  // generate a random transform for this instance
  
  vec3f boxCenter;
  vec3f rotationAxis;
  do {
    rotationAxis.x = rng();
    rotationAxis.y = rng();
    rotationAxis.z = rng();
  } while (dot(rotationAxis,rotationAxis) > 1.f);
  rotationAxis = normalize(rotationAxis);
  
  float rotationSpeed = .1f + rng() * .7f;
  float rotationAngle0 = rng() * 2.f * M_PI;

  const float worldSize = 1;
  const vec3f boxSize   = (2*.4f*worldSize)/vec3f(optixLaunchParams.numBoxes);
  const float animSpeed = 4.f;

  vec3i boxID = vec3i(instanceIndex % 100,(instanceIndex / 100) % 100,(instanceIndex / (100 * 100)) % 100);
  
  vec3f rel = (vec3f(boxID)+.5f) / vec3f(optixLaunchParams.numBoxes);
  boxCenter = vec3f(-worldSize) + (2.f*worldSize)*rel;

  const float angle  = rotationAngle0 + rotationSpeed*t;
  const linear3f rot = linear3f::rotate(rotationAxis,angle);
  affine3f tfm = affine3f(rot,boxCenter);

  i.transform[0] = tfm.l.vx.x; i.transform[1] = tfm.l.vx.y; i.transform[ 2] = tfm.l.vx.z; i.transform[ 3] = tfm.p.x;
  i.transform[4] = tfm.l.vy.x; i.transform[5] = tfm.l.vy.y; i.transform[ 6] = tfm.l.vy.z; i.transform[ 7] = tfm.p.y;
  i.transform[8] = tfm.l.vz.x; i.transform[9] = tfm.l.vz.y; i.transform[10] = tfm.l.vz.z; i.transform[11] = tfm.p.z;
}
