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

struct Hit {
  bool  hadHit = false;
  vec3f pos, nor, col;
};

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  Random rng(pixelID);
  
  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);
  ray.time = 0.5f;

  vec3f avgColor = 0.f;
  const int numSPP = 16;
  for (int i=0;i<numSPP;i++) {
    ray.time = rng();
    Hit hit;
    owl::traceRay(/*accel to trace against*/self.world,
                  /*the ray to trace*/ray,
                  /*prd*/hit);
    avgColor += hit.col;
  }
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(avgColor * (1.f/numSPP));
}

OPTIX_MOTION_BOUNDS_PROGRAM(Bounds)(const void *geomData,
                                    box3f &primBounds1,
                                    box3f &primBounds2,
                                    const int primID)
{
  const BoundsGeomData &self = *(const BoundsGeomData*)geomData;

  const vec3f center1 = self.vertex[primID * 2 + 0];
  const vec3f center2 = self.vertex[primID * 2 + 1];
  
  primBounds1.lower = center1 - vec3f(.1f, .1f, .1f);
  primBounds1.upper = center1 + vec3f(.1f, .1f, .1f);

  primBounds2.lower = center2 - vec3f(.1f, .1f, .1f);
  primBounds2.upper = center2 + vec3f(.1f, .1f, .1f);
}

OPTIX_INTERSECT_PROGRAM(Bounds)()
{
  const auto &self
    = owl::getProgramData<BoundsGeomData>();

  uint32_t primID = optixGetPrimitiveIndex();
  float time = optixGetRayTime();

  vec3f origin = optixGetObjectRayOrigin();
  vec3f direction = optixGetObjectRayDirection();
  float tmin = optixGetRayTmin();
  float tmax = optixGetRayTmax();

  const vec3f center1 = self.vertex[primID * 2 + 0];
  const vec3f center2 = self.vertex[primID * 2 + 1];

  vec3f center = center1 * (1.f - time) + center2 * time;
  box3f box = box3f(center - vec3f(.1f, .1f, .1f),
                    center + vec3f(.1f, .1f, .1f));

  float tmp = 0.f;
  optixReportIntersection(.1f,0,*(unsigned*)&tmp);

  // // intersect ray with box
  // vec3f invDir = 1.0f / direction;
  // vec3f t0 = (box.lower - origin) * invDir;
  // vec3f t1 = (box.upper - origin) * invDir;
  // vec3f tminVec = min(t0, t1);
  // vec3f tmaxVec = max(t0, t1);
  // float boxTmin = max(max(tminVec.x, tminVec.y), tminVec.z);
  // float boxTmax = min(min(tmaxVec.x, tmaxVec.y), tmaxVec.z);

  // // ray does not hit box
  // if (boxTmin > boxTmax) return;

  // // ray might hit box, tell optix to shorten ray
  // optixReportIntersection(boxTmin,0,*(unsigned*)&boxTmin);
}


__device__ vec3f hsvToRgb(const vec3f& hsv) {
  float h = hsv.x;
  float s = hsv.y;
  float v = hsv.z;

  float c = v * s;
  float x = c * (1.f - abs(fmodf(h / 60.0f, 2.f) - 1.f));
  float m = v - c;

  float r, g, b;
  if (h >= 0 && h < 60) {
    r = c;
    g = x;
    b = 0;
  } else if (h >= 60 && h < 120) {
    r = x;
    g = c;
    b = 0;
  } else if (h >= 120 && h < 180) {
    r = 0;
    g = c;
    b = x;
  } else if (h >= 180 && h < 240) {
    r = 0;
    g = x;
    b = c;
  } else if (h >= 240 && h < 300) {
    r = x;
    g = 0;
    b = c;
  } else {
    r = c;
    g = 0;
    b = x;
  }

  return vec3f(r + m, g + m, b + m);
}


OPTIX_CLOSEST_HIT_PROGRAM(BoundsMesh)()
{
  Hit &prd = owl::getPRD<Hit>();
  int primID = optixGetPrimitiveIndex();

  // Generate a vec3f color based on primID
  prd.col = hsvToRgb(vec3f((primID / 64.f) * 360.f, 1.f, 1.f)); 
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

