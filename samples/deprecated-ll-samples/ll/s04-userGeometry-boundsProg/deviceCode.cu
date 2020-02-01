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

#include "deviceCode.h"
#include <optix_device.h>

// not yet used in this example - ll03 still supplies bounds from the
// host through a buffer
OPTIX_BOUNDS_PROGRAM(Sphere)(const void  *geomData,
                             box3f       &primBounds,
                             const int    primID)
{
  const SphereGeomData &self = *(const SphereGeomData*)geomData;
  primBounds = box3f()
    .extend(self.center - self.radius)
    .extend(self.center + self.radius);
}

OPTIX_INTERSECT_PROGRAM(Sphere)()
{
  const SphereGeomData &self = owl::getProgramData<SphereGeomData>();
  // get index of primitive we are to intersect (for this example,
  // this will always be 0, because we have N differnent *geoms* with
  // one prim each.
  int primID = optixGetPrimitiveIndex();
  
  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  float hit_t      = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  const vec3f oc = org - self.center;
  const float a = dot(dir,dir);
  const float b = dot(oc, dir);
  const float c = dot(oc, oc) - self.radius * self.radius;
  const float discriminant = b * b - a * c;
  
  if (discriminant < 0.f) return;

  {
    float temp = (-b - sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
      
  {
    float temp = (-b + sqrtf(discriminant)) / a;
    if (temp < hit_t && temp > tmin) 
      hit_t = temp;
  }
  if (hit_t < optixGetRayTmax()) {
    optixReportIntersection(hit_t, 0);
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(Sphere)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const SphereGeomData &self = owl::getProgramData<SphereGeomData>();
  
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f Ng     = normalize(hit_P-self.center);

  prd = (.2f + .8f*fabs(dot(dir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
  if (pixelID.x >= self.fbSize.x) return;
  if (pixelID.y >= self.fbSize.y) return;

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  vec3f color;
  owl::trace(/*accel to trace against*/self.world,
             /*the ray to trace*/ ray,
             /*numRayTypes*/1,
             /*prd*/color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}


