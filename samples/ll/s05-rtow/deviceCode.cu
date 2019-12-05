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
OPTIX_BOUNDS_PROGRAM(LambertianSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{
  const LambertianSpheresGeom &self = *(const LambertianSpheresGeom*)geomData;
  const Sphere sphere = self.prims[primID].sphere;
  primBounds = box3f()
    .extend(sphere.center - sphere.radius)
    .extend(sphere.center + sphere.radius);
}

OPTIX_INTERSECT_PROGRAM(LambertianSpheres)()
{
  const int primID = optixGetPrimitiveIndex();
  const LambertianSphere &self
    = owl::getProgramData<LambertianSpheresGeom>().prims[primID];
  
  const vec3f org  = optixGetWorldRayOrigin();
  const vec3f dir  = optixGetWorldRayDirection();
  float hit_t      = optixGetRayTmax();
  const float tmin = optixGetRayTmin();

  const vec3f oc = org - self.sphere.center;
  const float a = dot(dir,dir);
  const float b = dot(oc, dir);
  const float c = dot(oc, oc) - self.sphere.radius * self.sphere.radius;
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

OPTIX_CLOSEST_HIT_PROGRAM(LambertianSpheres)()
{
  const int primID = optixGetPrimitiveIndex();
  const LambertianSphere &self
    = owl::getProgramData<LambertianSpheresGeom>().prims[primID];
  
  vec3f &prd = owl::getPRD<vec3f>();

  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f Ng    = normalize(hit_P-self.sphere.center);

  prd = (.2f + .8f*fabs(dot(dir,Ng)))*self.lambertian.albedo;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();

  const vec3f unit_direction = normalize((vec3f)optixGetWorldRayDirection());
  const float t = 0.5f*(unit_direction.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  vec3f &prd = owl::getPRD<vec3f>();
  prd = c;
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
  if (pixelID.x >= self.fbSize.x) return;
  if (pixelID.y >= self.fbSize.y) return;

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;

  // const vec3f rd = camera_lens_radius * random_in_unit_disk(rnd);
  // const vec3f lens_offset = camera_u * rd.x + camera_v * rd.y;
  const vec3f origin = self.camera.origin // + lens_offset
    ;
  const vec3f direction
    = self.camera.lower_left_corner
    + screen.u * self.camera.horizontal
    + screen.v * self.camera.vertical
    - self.camera.origin;
  
  ray.origin = origin;
  ray.direction = direction;

  vec3f color;
  owl::trace(/*accel to trace against*/self.world,
             /*the ray to trace*/ ray,
             /*numRayTypes*/1,
             /*prd*/color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*(self.fbSize.y-1-pixelID.y);
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}


