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

#define NUM_SAMPLES_PER_PIXEL 128

inline __device__
float schlick(float cosine,
              float ref_idx)
{
  float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
  r0 = r0 * r0;
  return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
}

inline __device__
bool refract(const vec3f& v,
             const vec3f& n,
             float ni_over_nt,
             vec3f &refracted)
{
  vec3f uv = normalize(v);
  float dt = dot(uv, n);
  float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
  if (discriminant > 0.f) {
    refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
    return true;
  }
  else
    return false;
}

inline __device__
vec3f reflect(const vec3f &v,
              const vec3f &n)
{
  return v - 2.0f*dot(v, n)*n;
}

struct DRand48
{
  /*! initialize the random number generator with a new seed (usually
      per pixel) */
  inline __device__ void init(int seed = 0)
  {
    state = seed;
    for (int warmUp=0;warmUp<10;warmUp++)
      (*this)();
  }

  /*! get the next 'random' number in the sequence */
  inline __device__ float operator() ()
  {
    const uint64_t a = 0x5DEECE66DULL;
    const uint64_t c = 0xBULL;
    const uint64_t mask = 0xFFFFFFFFFFFFULL;
    state = a*state + c;
    return float((state & mask) / float(mask+1ULL));
  }

  uint64_t state;
};



typedef enum {
  /*! ray could get properly bounced, and is still alive */
  rayGotBounced,
  /*! ray could not get scattered, and should get cancelled */
  rayGotCancelled,
  /*! ray didn't hit anything, and went into the environemnt */
  rayDidntHitAnything
} ScatterEvent;

/*! "per ray data" (PRD) for our sample's rays. In the simple example, there is only
  one ray type, and it only ever returns one thing, which is a color (everything else
  is handled through the recursion). In addition to that return type, rays have to
  carry recursion state, which in this case are recursion depth and random number state */
struct PerRayData {
  DRand48 random;
  struct {
    ScatterEvent scatterEvent;
    vec3f        scattered_origin;
    vec3f        scattered_direction;
    vec3f        attenuation;
  } out;
};




inline __device__ vec3f randomPointOnUnitDisc(DRand48 &random) {
  vec3f p;
  do {
    p = 2.0f*vec3f(random(), random(), 0.f) - vec3f(1.f, 1.f, 0.f);
  } while (dot(p, p) >= 1.0f);
  return p;
}


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

inline __device__ vec3f randomPointInUnitSphere(DRand48 &rnd) {
  vec3f p;
  do {
    p = 2.0f*RANDVEC3F - vec3f(1, 1, 1);
  } while (dot(p,p) >= 1.0f);
  return p;
}


inline __device__
bool scatter(const Lambertian &lambertian,
             const vec3f &P,
             vec3f N,
             // const owl::Ray &ray_in,
             PerRayData &prd)
{
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();

  if (dot(N,dir)  > 0.f)
    N = -N;
  N = normalize(N);

  const vec3f target
    = P + (N + randomPointInUnitSphere(prd.random));

  
  // return scattering event
  prd.out.scattered_origin    = P;
  prd.out.scattered_direction = (target-P);
  prd.out.attenuation         = lambertian.albedo;
  return true;
}

inline __device__
bool scatter(const Dielectric &dielectric,
             const vec3f &P,
             vec3f N,
             PerRayData &prd)
{
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = normalize((vec3f)optixGetWorldRayDirection());

  N = normalize(N);
  vec3f outward_normal;
  vec3f reflected = reflect(dir,N);
  float ni_over_nt;
  prd.out.attenuation = vec3f(1.f, 1.f, 1.f); 
  vec3f refracted;
  float reflect_prob;
  float cosine;
  
  if (dot(dir,N) > 0.f) {
    outward_normal = -N;
    ni_over_nt = dielectric.ref_idx;
    cosine = dot(dir, N);// / vec3f(dir).length();
    cosine = sqrtf(1.f - dielectric.ref_idx*dielectric.ref_idx*(1.f-cosine*cosine));
  }
  else {
    outward_normal = N;
    ni_over_nt = 1.0 / dielectric.ref_idx;
    cosine = -dot(dir, N);// / vec3f(dir).length();
  }
  if (refract(dir, outward_normal, ni_over_nt, refracted)) 
    reflect_prob = schlick(cosine, dielectric.ref_idx);
  else 
    reflect_prob = 1.f;

  prd.out.scattered_origin = P;
  if (prd.random() < reflect_prob) 
    prd.out.scattered_direction = reflected;
  else 
    prd.out.scattered_direction = refracted;
  
  return true;
}

inline __device__
bool scatter(const Metal &metal,
             const vec3f &P,
             vec3f N,
             PerRayData &prd)
{
  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();

  if (dot(N,dir)  > 0.f)
    N = -N;
  N = normalize(N);
  
  vec3f reflected = reflect(normalize(dir),N);
  prd.out.scattered_origin    = P;
  prd.out.scattered_direction
    = (reflected+metal.fuzz*randomPointInUnitSphere(prd.random));
  prd.out.attenuation         = metal.albedo;
  return (dot(prd.out.scattered_direction, N) > 0.f);
}



// ==================================================================
// bounding box programs - since these don't actually use the material
// they're all the same irrespective of geometry type, so use a
// template ...
// ==================================================================
template<typename SphereGeomType>
inline __device__ void boundsProg(const void *geomData,
                                  box3f &primBounds,
                                  const int primID)
{
  const SphereGeomType &self = *(const SphereGeomType*)geomData;
  const Sphere sphere = self.prims[primID].sphere;
  primBounds = box3f()
    .extend(sphere.center - sphere.radius)
    .extend(sphere.center + sphere.radius);
}

OPTIX_BOUNDS_PROGRAM(MetalSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<MetalSpheresGeom>(geomData,primBounds,primID); }

OPTIX_BOUNDS_PROGRAM(LambertianSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<LambertianSpheresGeom>(geomData,primBounds,primID); }

OPTIX_BOUNDS_PROGRAM(DielectricSpheres)(const void  *geomData,
                                        box3f       &primBounds,
                                        const int    primID)
{ boundsProg<DielectricSpheresGeom>(geomData,primBounds,primID); }


// ==================================================================
// intersect programs - still all the same, since they don't use the
// material, either
// ==================================================================

template<typename SpheresGeomType>
inline __device__ void intersectProg()
{
  const int primID = optixGetPrimitiveIndex();
  const auto &self
    = owl::getProgramData<SpheresGeomType>().prims[primID];
  
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


OPTIX_INTERSECT_PROGRAM(MetalSpheres)()
{ intersectProg<MetalSpheresGeom>(); }

OPTIX_INTERSECT_PROGRAM(LambertianSpheres)()
{ intersectProg<LambertianSpheresGeom>(); }

OPTIX_INTERSECT_PROGRAM(DielectricSpheres)()
{ intersectProg<DielectricSpheresGeom>(); }


// ==================================================================
// plumbing for closest hit
// ==================================================================

template<typename SpheresGeomType>
inline __device__
void closestHit()
{
  const int primID = optixGetPrimitiveIndex();
  const auto &self
    = owl::getProgramData<SpheresGeomType>().prims[primID];
  
  PerRayData &prd = owl::getPRD<PerRayData>();

  const vec3f org   = optixGetWorldRayOrigin();
  const vec3f dir   = optixGetWorldRayDirection();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f N     = (hit_P-self.sphere.center);

  prd.out.scatterEvent
    = scatter(self.material,
              hit_P,N,//ray,
              prd)
    ? rayGotBounced
    : rayGotCancelled;
}

OPTIX_CLOSEST_HIT_PROGRAM(MetalSpheres)()
{ closestHit<MetalSpheresGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(LambertianSpheres)()
{ closestHit<LambertianSpheresGeom>(); }
OPTIX_CLOSEST_HIT_PROGRAM(DielectricSpheres)()
{ closestHit<DielectricSpheresGeom>(); }









// ==================================================================
// miss and raygen
// ==================================================================

inline __device__
vec3f missColor(const Ray &ray)
{
  const vec2i pixelID = owl::getLaunchIndex();

  const vec3f rayDir = normalize(ray.direction);
  const float t = 0.5f*(rayDir.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
  PerRayData &prd = owl::getPRD<PerRayData>();
  prd.out.scatterEvent = rayDidntHitAnything;
  // const vec2i pixelID = owl::getLaunchIndex();

  // const MissProgData &self = owl::getProgramData<MissProgData>();

  // const vec3f unit_direction = normalize((vec3f)optixGetWorldRayDirection());
  // const float t = 0.5f*(unit_direction.y + 1.0f);
  // const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  // vec3f &prd = owl::getPRD<vec3f>();
  // prd = c;
}



inline __device__
vec3f tracePath(const RayGenData &self,
                owl::Ray &ray, PerRayData &prd)
{
  vec3f attenuation = 1.f;
  
  /* iterative version of recursion, up to depth 50 */
  for (int depth=0;depth<50;depth++) {
    owl::trace(/*accel to trace against*/self.world,
               /*the ray to trace*/ ray,
               /*numRayTypes*/1,
               /*prd*/prd);
    
    if (prd.out.scatterEvent == rayDidntHitAnything)
      /* ray got 'lost' to the environment - 'light' it with miss
         shader */
      return attenuation * missColor(ray);
    else if (prd.out.scatterEvent == rayGotCancelled)
      return vec3f(0.f);

    else { // ray is still alive, and got properly bounced
      attenuation *= prd.out.attenuation;
      ray = owl::Ray(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ prd.out.scattered_direction,
                     /* ray type : */ 0,
                     /* tmin     : */ 1e-3f,
                     /* tmax     : */ 1e10f);
    }
  }
  // recursion did not terminate - cancel it
  return vec3f(0.f);
}

OPTIX_RAYGEN_PROGRAM(rayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  
  if (pixelID.x >= self.fbSize.x) return;
  if (pixelID.y >= self.fbSize.y) return;
  const int pixelIdx = pixelID.x+self.fbSize.x*(self.fbSize.y-1-pixelID.y);

  // for multi-gpu: only render every deviceCount'th column of 32 pixels:
  if (((pixelID.x/32) % self.deviceCount) != self.deviceIndex)
    return;
  
  PerRayData prd;
  prd.random.init(pixelIdx);
  
  vec3f color = 0.f;
  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {
    owl::Ray ray;
    
    const vec2f pixelSample(prd.random(),prd.random());
    const vec2f screen
      = (vec2f(pixelID)+pixelSample)
      / vec2f(self.fbSize);
    const vec3f origin = self.camera.origin // + lens_offset
      ;
    const vec3f direction
      = self.camera.lower_left_corner
      + screen.u * self.camera.horizontal
      + screen.v * self.camera.vertical
      - self.camera.origin;
  
    ray.origin = origin;
    ray.direction = direction;

    color += tracePath(self, ray, prd);
  }
    
  self.fbPtr[pixelIdx]
    = owl::make_rgba(color * (1.f / NUM_SAMPLES_PER_PIXEL));
}


