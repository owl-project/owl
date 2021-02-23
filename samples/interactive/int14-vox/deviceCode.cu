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
#include <owl/common/math/LinearSpace.h>

__constant__ LaunchParams optixLaunchParams;


typedef owl::common::LCG<4> Random;
typedef owl::RayT<0, 2> RadianceRay;
typedef owl::RayT<1, 2> ShadowRay;

inline __device__
vec3f missColor(const RadianceRay &ray)
{
  const vec2i pixelID = owl::getLaunchIndex();

  const vec3f rayDir = normalize(ray.direction);
  const float t = 0.5f*(rayDir.y + 1.0f);
  const vec3f c = (1.0f - t)*vec3f(1.0f, 1.0f, 1.0f) + t * vec3f(0.5f, 0.7f, 1.0f);
  return c;
}

OPTIX_MISS_PROGRAM(miss)()
{
  /* nothing to do */
}

OPTIX_MISS_PROGRAM(miss_shadow)()
{
  float &vis = owl::getPRD<float>();
  vis = 1.f;
}

typedef enum {
  /*! ray could get properly bounced, and is still alive */
  rayGotBounced,
  /*! ray could not get scattered, and should get cancelled */
  rayGotCancelled,
  /*! ray didn't hit anything, and went into the environment */
  rayDidntHitAnything
} ScatterEvent;


struct PerRayData
{
  Random random;
  struct {
    ScatterEvent scatterEvent;
    vec3f        scattered_origin;
    vec3f        scattered_direction;
    vec3f        attenuation;
    vec3f        directLight;
  } out;
};

inline __device__
vec3f tracePath(const RayGenData &self,
                RadianceRay &ray, PerRayData &prd)
{
  vec3f attenuation = 1.f;
  vec3f directLight = 0.0f;
  
  /* iterative version of recursion, up to max depth */
  for (int depth=0;depth<1;depth++) {
    prd.out.scatterEvent = rayDidntHitAnything;
    owl::traceRay(/*accel to trace against*/ optixLaunchParams.world,
                  /*the ray to trace*/ ray,
                  /*prd*/prd);

    
    if (prd.out.scatterEvent == rayDidntHitAnything)
      /* ray got 'lost' to the environment - 'light' it with miss
         shader */
      return directLight + attenuation * missColor(ray);
    else if (prd.out.scatterEvent == rayGotCancelled)
      return vec3f(0.f);

    else { // ray is still alive, and got properly bounced
      attenuation *= prd.out.attenuation;
      directLight += prd.out.directLight;
      ray = RadianceRay(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ prd.out.scattered_direction,
                     /* tmin     : */ 1e-3f,
                     /* tmax     : */ 1e10f);
    }
  }
  // recursion did not terminate - cancel it but return direct lighting from any previous bounce
  return directLight;
}

// returns a visibility term (1 for unshadowed)
inline __device__
float traceShadowRay(const OptixTraversableHandle &traversable,
                  ShadowRay &ray)
{
  float vis = 0.f;
  owl::traceRay(traversable, ray, vis, 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT
                   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return vis;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  PerRayData prd;
  prd.random.init(pixelID.x,pixelID.y);

  const int NUM_SAMPLES_PER_PIXEL = 12;  // TODO: jitter?
  vec3f accumColor = 0.f;

  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {
    RadianceRay ray;

    const vec2f pixelSample(prd.random(), prd.random());
    const vec2f screen = (vec2f(pixelID)+pixelSample) / vec2f(self.fbSize);

    ray.origin = self.camera.pos;
    ray.direction 
      = normalize(self.camera.dir_00
                  + screen.u * self.camera.dir_du
                  + screen.v * self.camera.dir_dv);

    accumColor += tracePath(self, ray, prd);
  }
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(accumColor * (1.f / NUM_SAMPLES_PER_PIXEL));
}

// from OptiX 6 SDK
#ifndef M_PI_4f
#define M_PI_4f     0.785398163397448309616f
#endif
inline __device__ 
float2 squareToDisk(const float2& sample)
{
  float phi, r;

  const float a = 2.0f * sample.x - 1.0f;
  const float b = 2.0f * sample.y - 1.0f;

  if (a > -b)
  {
    if (a > b)
    {
      r = a;
      phi = (float)M_PI_4f * (b/a);
    }
    else
    {
      r = b;
      phi = (float)M_PI_4f * (2.0f - (a/b));
    }
  }
  else
  {
    if (a < b)
    {
      r = -a;
      phi = (float)M_PI_4f * (4.0f + (b/a));
    }
    else
    {
      r = -b;
      phi = (b) ? (float)M_PI_4f * (6.0f - (a/b)) : 0.0f;
    }
  }

  return make_float2( r * cosf(phi), r * sinf(phi) );
}

inline __device__
vec3f sunlight(const vec3f &Ng, float shadowBias, Random &random)
{
  // Get direct light
  const vec3f dir   = optixGetWorldRayDirection();
  const vec3f org   = optixGetWorldRayOrigin();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f hit_P_offset = hit_P + shadowBias*Ng;  // bias along normal to help with shadow acne
  const vec3f lightDir = optixLaunchParams.sunDirection;

  // Build frame around light dir
  const owl::LinearSpace3f lightFrame = owl::common::frame(normalize(lightDir));

  // jitter light direction slightly
  const float lightRadius = 0.01f;  // should be ok as a constant since our scenes are normalized
  const vec3f lightCenter = hit_P_offset + lightFrame.vz;
  const float2 sample = squareToDisk(make_float2(random(), random()));
  const vec3f jitteredPos = lightCenter + lightRadius*(sample.x*lightFrame.vx + sample.y*lightFrame.vy);
  const vec3f jitteredLightDir = jitteredPos - hit_P_offset;  // no need to normalize

  ShadowRay shadowRay(hit_P_offset,      // origin
                      jitteredLightDir,  // direction
                      shadowBias,
                      1e10f);

  float vis = traceShadowRay(optixLaunchParams.world, shadowRay);
  return vis * optixLaunchParams.sunColor;

}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Nbox   = normalize(cross(B-A,C-A));
  const vec3f Ng     = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));
  const float boxScaleInWorldSpace = owl::length(vec3f(optixTransformVectorFromObjectToWorldSpace(make_float3(0,0,1))));

  // Bias value relative to a brick; handle large bricks with a clamp
  const float shadowBias = 1e-2f * fminf(1.f, boxScaleInWorldSpace);

  // Convert 8 bit color to float
  const unsigned int instanceID = optixGetInstanceId();
  const int ci = self.colorIndexPerInstance[instanceID];
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  PerRayData &prd = owl::getPRD<PerRayData>();
  const vec3f directLight = sunlight(Ng, shadowBias, prd.random);
  prd.out.directLight = directLight*color;  //debug

  // Bounce
  prd.out.attenuation = color;
  prd.out.scatterEvent = rayGotBounced;

}

OPTIX_BOUNDS_PROGRAM(VoxGeom)(const void *geomData,
                              box3f &primBounds,
                              const int primID)
{
  const VoxGeomData &self = *(const VoxGeomData*)geomData;
  uchar4 indices = self.prims[primID];
  const vec3f boxmin( indices.x, indices.y, indices.z );
  const vec3f boxmax( 1+indices.x, 1+indices.y, 1+indices.z );
  primBounds = box3f(boxmin, boxmax);
}

inline __device__ int indexOfMaxComponent(vec3f v)
{
  if (v.x > v.y) 
    return v.x > v.z ? 0 : 2;
  else
    return v.y > v.z ? 1 : 2;
}

OPTIX_INTERSECT_PROGRAM(VoxGeom)()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();
  uchar4 indices = self.prims[primID];
  vec3f boxCenter(indices.x+0.5, indices.y+0.5, indices.z+0.5);

  // Translate ray to local box space
  const vec3f rayOrigin  = vec3f(optixGetObjectRayOrigin()) - boxCenter;
  const vec3f rayDirection  = optixGetObjectRayDirection();  // assume no rotation
  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  const float ray_tmax = optixGetRayTmax();
  const float ray_tmin = optixGetRayTmin();

  const vec3f boxRadius(0.5f, 0.5f, 0.5f);
  vec3f t0 = (-boxRadius - rayOrigin) * invRayDirection;
  vec3f t1 = ( boxRadius - rayOrigin) * invRayDirection;
  float tnear = reduce_max(owl::min(t0, t1));
  float tfar  = reduce_min(owl::max(t0, t1));

  // Only handle the case where the ray starts outside the box

  if (tnear <= tfar && tnear > ray_tmin && tnear < ray_tmax) {
    // compute face normal at local hit point
    vec3f V = rayOrigin + rayDirection*tnear;
    vec3f N(0.0f);
    int i = indexOfMaxComponent(abs(V));
    N[i] = (V[i] >= 0.0f) ? 1 : -1;
    optixReportIntersection( tnear, 0, float_as_int(N.x), float_as_int(N.y), float_as_int(N.z));
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(VoxGeom)()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();

  // Select normal for whichever face we hit
  const float3 Nbox = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2()));
  const vec3f Ng = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));
  const float boxScaleInWorldSpace = owl::length(vec3f(optixTransformVectorFromObjectToWorldSpace(make_float3(0,0,1))));

  // Bias value relative to a brick; handle large bricks with a clamp
  const float shadowBias = 1e-2f * fminf(1.f, boxScaleInWorldSpace);

  // Convert 8 bit color to float
  const int ci = self.prims[primID].w;
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  PerRayData &prd = owl::getPRD<PerRayData>();
  const vec3f directLight = sunlight(Ng, shadowBias, prd.random);
  prd.out.directLight = directLight*color;  //debug

  // Bounce
  prd.out.attenuation = color;
  prd.out.scatterEvent = rayGotBounced;

}


