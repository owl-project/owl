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
#include "constants.h"
#include <optix_device.h>

#include <owl/common/math/random.h>
#include <owl/common/math/LinearSpace.h>


__constant__ LaunchParams optixLaunchParams;


typedef owl::common::LCG<4> Random;
typedef owl::RayT<0, 3> RadianceRay;
typedef owl::RayT<1, 3> ShadowRay;
typedef owl::RayT<2, 3> OutlineShadowRay;

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
#if ENABLE_TOON_OUTLINE
    float        hitDistance;
#endif
  } out;
};

inline __device__
RadianceRay makeRadianceRay(const vec3f &origin, 
                            const vec3f &direction,
                            float tmin,
                            float tmax)
{
  return RadianceRay(origin,
                     direction,
                     tmin,
                     tmax,
                     VISIBILITY_RADIANCE);
}


inline __device__
vec3f tracePath(const RayGenData &self,
                RadianceRay &ray, PerRayData &prd, float &firstHitDistance)
{
  vec3f attenuation = 1.f;
  vec3f directLight = 0.0f;
#if ENABLE_TOON_OUTLINE
  firstHitDistance = 1e10f;
#endif

  constexpr int MaxDepth = 2;
  
  /* iterative version of recursion, up to max depth */
  for (int depth=0;depth<MaxDepth;depth++) {
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
      ray = makeRadianceRay(/* origin   : */ prd.out.scattered_origin,
                     /* direction: */ prd.out.scattered_direction,
                     /* tmin     : */ 1e-3f,
                     /* tmax     : */ 1e10f);
#if ENABLE_TOON_OUTLINE
      if (depth == 0) {
        firstHitDistance = prd.out.hitDistance;
      }
#endif
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

// returns a visibility term (1 for unshadowed)
inline __device__
float traceOutlineShadowRay(const OptixTraversableHandle &traversable,
                  OutlineShadowRay &ray)
{
  float vis = 0.f;
  owl::traceRay(traversable, ray, vis, 
                   OPTIX_RAY_FLAG_DISABLE_ANYHIT
                   | OPTIX_RAY_FLAG_CULL_FRONT_FACING_TRIANGLES
                   | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT
                   | OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT);
  return vis;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  const vec2i fbSize = optixLaunchParams.fbSize;
  const int fbIndex = pixelID.x+fbSize.x*pixelID.y;

  PerRayData prd;
  prd.random.init(fbIndex, optixLaunchParams.frameID);

  const int NUM_SAMPLES_PER_PIXEL = 4;
  vec3f accumColor = 0.f;

  for (int sampleID=0;sampleID<NUM_SAMPLES_PER_PIXEL;sampleID++) {

    const vec2f pixelSample(prd.random(), prd.random());
    const vec2f screen = (vec2f(pixelID)+pixelSample) / vec2f(fbSize);

    const vec3f rayDir = normalize(self.camera.dir_00
                  + screen.u * self.camera.dir_du
                  + screen.v * self.camera.dir_dv);

    RadianceRay ray = makeRadianceRay(self.camera.pos, rayDir, 0.f, 1e30f);

    // needed later for outline shader
    float firstHitDistance = 1e10f;

    vec3f color = tracePath(self, ray, prd, firstHitDistance);

#if ENABLE_TOON_OUTLINE
    if (1 /*optixLaunchParams.enableToonOutline*/ ) {
      constexpr float outlineDepthBias = 0.05f;  // TODO: set from launch param
      OutlineShadowRay outlineShadowRay(self.camera.pos,      // origin
                      rayDir,  // same direction as eye ray
                      0.f,
                      firstHitDistance-outlineDepthBias,  // only show outline if it is in front of regular hit
                      VISIBILITY_OUTLINE);

      float vis = traceOutlineShadowRay(optixLaunchParams.world, outlineShadowRay);
      color *= vis;
    }
#endif // ENABLE_TOON_OUTLINE

    accumColor += color;
  }
    
  vec4f rgba {accumColor / NUM_SAMPLES_PER_PIXEL, 1.0f};

  if (optixLaunchParams.frameID > 0) {
    // Blend with accum buffer
    const vec4f accum = optixLaunchParams.fbAccumBuffer[fbIndex];
    rgba += float(optixLaunchParams.frameID) * accum; 
    rgba /= (optixLaunchParams.frameID+1.f);
  }

  optixLaunchParams.fbAccumBuffer[fbIndex] = (float4)rgba;
  optixLaunchParams.fbPtr[fbIndex] = owl::make_rgba(rgba);
}

inline __device__ 
float2 squareToDisk(float u1, float u2)
{
  // Uniformly sample disk.
  const float r   = sqrtf( u1 );
  const float phi = 2.0f*M_PIf * u2;
  float2 p = {r * cosf( phi ), r * sinf( phi )};
  return p;
}

inline __device__
vec3f cosineSampleHemisphere(float u1, float u2)
{
  float2 p = squareToDisk(u1, u2);

  // Project up to hemisphere.
  return vec3f(p.x, p.y, sqrtf( fmaxf( 0.0f, 1.0f - p.x*p.x - p.y*p.y ) ));
}

inline __device__
vec3f sunlight(const vec3f &hit_P_offset, const vec3f &Ng, Random &random)
{
  const vec3f lightDir = optixLaunchParams.sunDirection;

  const float NdotL = dot(Ng, lightDir);
  if (NdotL <= 0.f) {
    return 0.f;  // below horizon
  }

  // Build frame around light dir
  const owl::LinearSpace3f lightFrame = owl::common::frame(normalize(lightDir));

  // jitter light direction slightly
  const float lightRadius = 0.01f;  // should be ok as a constant since our scenes are normalized
  const vec3f lightCenter = hit_P_offset + lightFrame.vz;
  const float2 sample = squareToDisk(random(), random());
  const vec3f jitteredPos = lightCenter + lightRadius*(sample.x*lightFrame.vx + sample.y*lightFrame.vy);
  const vec3f jitteredLightDir = jitteredPos - hit_P_offset;  // no need to normalize

  ShadowRay shadowRay(hit_P_offset,      // origin
                      jitteredLightDir,  // direction
                      0.f,
                      1e10f,
                      VISIBILITY_SHADOW);

  float vis = traceShadowRay(optixLaunchParams.world, shadowRay);
  return vis * optixLaunchParams.sunColor * NdotL; 

}

inline __device__ 
vec3f scatterLambertian(const vec3f &Ng, Random &random)
{
  const owl::LinearSpace3f shadingFrame = owl::common::frame(Ng);
  vec3f scatteredDirectionInShadingFrame = cosineSampleHemisphere(random(), random());
  vec3f scatteredDirection = shadingFrame.vx * scatteredDirectionInShadingFrame.x +
                             shadingFrame.vy * scatteredDirectionInShadingFrame.y +
                             shadingFrame.vz * scatteredDirectionInShadingFrame.z;

  return scatteredDirection;
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
  const unsigned int brickID = self.isFlat ? optixGetPrimitiveIndex() / self.primCountPerBrick : optixGetInstanceId();
  const int ci = self.colorIndexPerBrick[brickID];
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  PerRayData &prd = owl::getPRD<PerRayData>();
  const vec3f dir   = optixGetWorldRayDirection();
  const vec3f org   = optixGetWorldRayOrigin();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f hit_P_offset = hit_P + shadowBias*Ng;  // bias along normal to help with shadow acne

  // Direct
  const vec3f directLight = sunlight(hit_P_offset, Ng, prd.random);

  // Bounce
  vec3f scatteredDirection = scatterLambertian(Ng, prd.random);

  prd.out.directLight = directLight*color;
  prd.out.attenuation = color;
  prd.out.scatterEvent = rayGotBounced;
  prd.out.scattered_direction = scatteredDirection;
  prd.out.scattered_origin = hit_P_offset;
#if ENABLE_TOON_OUTLINE
  prd.out.hitDistance = length(hit_P - org);
#endif

}

OPTIX_BOUNDS_PROGRAM(VoxGeom)(const void *geomData,
                              box3f &primBounds,
                              const int primID)
{
  const VoxGeomData &self = *(const VoxGeomData*)geomData;
  uchar4 indices = self.prims[primID];
  vec3f boxmin( indices.x, indices.y, indices.z );
  vec3f boxmax( 1+indices.x, 1+indices.y, 1+indices.z );

#if ENABLE_TOON_OUTLINE
  if (self.enableToonOutline) {
    // bloat the box slightly
    const float scale = 1.2f;
    const vec3f boxcenter (indices.x + 0.5f, indices.y + 0.5f, indices.z + 0.5f);
    boxmin = boxcenter + scale*(boxmin-boxcenter);
    boxmax = boxcenter + scale*(boxmax-boxcenter);
  }
#endif
  
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

// Used for toon outline 
OPTIX_INTERSECT_PROGRAM(VoxGeomShadowCullFront)()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();
  if (!self.enableToonOutline) {
      return;
  }
  uchar4 indices = self.prims[primID];
  vec3f boxCenter(indices.x+0.5, indices.y+0.5, indices.z+0.5);

  // Translate ray to local box space
  const vec3f rayOrigin  = vec3f(optixGetObjectRayOrigin()) - boxCenter;
  const vec3f rayDirection  = optixGetObjectRayDirection();  // assume no rotation
  const vec3f invRayDirection = vec3f(1.0f) / rayDirection;

  const float ray_tmax = optixGetRayTmax();
  const float ray_tmin = optixGetRayTmin();

  const float outlinePad = 1.2f;  // needs to match bounding box program
  const vec3f boxRadius = vec3f(0.5f, 0.5f, 0.5f)*outlinePad;
  vec3f t0 = (-boxRadius - rayOrigin) * invRayDirection;
  vec3f t1 = ( boxRadius - rayOrigin) * invRayDirection;
  float tnear = reduce_max(owl::min(t0, t1));
  float tfar  = reduce_min(owl::max(t0, t1));

  // Cull front face by using tfar for the hit

  if (tnear <= tfar && tfar > ray_tmin && tfar < ray_tmax) {
    optixReportIntersection( tfar, 0);
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
  const vec3f dir   = optixGetWorldRayDirection();
  const vec3f org   = optixGetWorldRayOrigin();
  const float hit_t = optixGetRayTmax();
  const vec3f hit_P = org + hit_t * dir;
  const vec3f hit_P_offset = hit_P + shadowBias*Ng;  // bias along normal to help with shadow acne

  // Direct
  const vec3f directLight = sunlight(hit_P_offset, Ng, prd.random);

  // Bounce
  vec3f scatteredDirection = scatterLambertian(Ng, prd.random);

  prd.out.directLight = directLight*color;
  prd.out.attenuation = color;
  prd.out.scatterEvent = rayGotBounced;
  prd.out.scattered_direction = scatteredDirection;
  prd.out.scattered_origin = hit_P_offset;
#if ENABLE_TOON_OUTLINE
  prd.out.hitDistance = length(hit_P - org);
#endif

}


