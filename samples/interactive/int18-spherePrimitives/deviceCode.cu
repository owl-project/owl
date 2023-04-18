// ======================================================================== //
// Copyright 2019-2022 Ingo Wald                                            //
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

/* this sample uses features from newer versions of optix; it will not
   run with older versions, nor even compile with them - so to make
   the entire project compile with older versions of optix we'll here
   disable this newer code if an older version is being used. */
#if OPTIX_VERSION >= 70500

#include "../int15-cookBilliardScene/helpers.h"
#include <optix_device.h>
#include <owl/common/math/random.h>

extern "C" __constant__ LaunchParams optixLaunchParams;

typedef owl::common::LCG<4> Random;

inline __device__ bool dbg()
{
  const vec2i launchIndex = owl::getLaunchIndex();
  const vec2i launchDims  = owl::getLaunchDims();
  const bool dbg_x = launchIndex.x == launchDims.x/2;
  const bool dbg_y = launchIndex.y == launchDims.y/2;
  return dbg_x & dbg_y;
}

struct PRD {
  Random rng;
  float t_hit;
  vec3f gn, sn;
  struct {
    vec3f result;
    float importance;
    int depth;
  } radiance;
  struct {
    vec3f attenuation;
  } shadow;
  int max_depth;
};

static
__device__ void phongShade( vec3f p_Kd,
                            vec3f p_Ka,
                            vec3f p_Ks,
                            vec3f p_normal,
                            float p_phong_exp,
                            vec3f p_reflectivity )
{
  const auto &self
    = owl::getProgramData<SpheresGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  vec3f hit_point = ray.origin + prd.t_hit * ray.direction;
  
  // ambient contribution
  vec3f result = p_Ka * optixLaunchParams.ambient_light_color;

  // compute direct lighting
  unsigned int num_lights = optixLaunchParams.numLights;

  for(int i = 0; i < num_lights; ++i) {
    // set jittered light direction
    BasicLight light = optixLaunchParams.lights[i];
    vec3f L = light.pos - hit_point;

    vec2f sample = square_to_disk(vec2f(prd.rng(),prd.rng()));
    vec3f U, V, W;
    create_onb(L, U, V, W);
    L += 5.0f * (sample.x * U + sample.y * V);

    float Ldist = length(L);
    L = (1.0f / Ldist) * L;

    float nDl = dot( p_normal, L);

    // cast shadow ray
    PRD shadow_prd;
    shadow_prd.shadow.attenuation = vec3f(1.f);
    if(nDl > 0) {
      ShadowRay shadow_ray(hit_point,L,optixLaunchParams.scene_epsilon,Ldist);
      owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
                    /*the ray to trace*/shadow_ray,
                    /*prd*/shadow_prd);
    }

    // If not completely shadowed, light the hit point
    if(fmaxf(shadow_prd.shadow.attenuation) > 0) {
      vec3f Lc = light.color * shadow_prd.shadow.attenuation;

      result += p_Kd * nDl * Lc;

      vec3f H = normalize(L - ray.direction);
      float nDh = dot( p_normal, H );
      if(nDh > 0) {
        float power = pow(nDh, p_phong_exp);
        result += p_Ks * power * Lc;
      }
    }
  }

  if( fmaxf( p_reflectivity ) > 0 ) {

    // ray tree attenuation
    PRD new_prd;             
    vec3f ntsc_luminance = {0.30, 0.59, 0.11}; 
    new_prd.radiance.importance = prd.radiance.importance * dot( p_reflectivity, ntsc_luminance );
    new_prd.radiance.depth = prd.radiance.depth + 1;

    // reflection ray
    if( new_prd.radiance.importance >= 0.01f && new_prd.radiance.depth <= prd.max_depth) {
      vec3f R = reflect( ray.direction, p_normal );

      RadianceRay refl_ray(hit_point,R,optixLaunchParams.scene_epsilon,1e30f);
      owl::traceRay(/*accel to trace against*/optixLaunchParams.world,
                    /*the ray to trace*/refl_ray,
                    /*prd*/new_prd,
                    /*only CH*/OPTIX_RAY_FLAG_DISABLE_ANYHIT);
      result += p_reflectivity * new_prd.radiance.result;
    }
  }

  // pass the color back up the tree
  prd.radiance.result = result;
}

OPTIX_CLOSEST_HIT_PROGRAM(SpheresGeom)()
{
  const auto &self
    = owl::getProgramData<SpheresGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  prd.t_hit = optixGetRayTmax(); // TODO:

  // Taken from Optix 7.6.0 SDK/optixSphere
  const unsigned int           prim_idx    = optixGetPrimitiveIndex();
  const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
  const unsigned int           sbtGASIndex = optixGetSbtGASIndex();
  float4 q;
  // sphere center (q.x, q.y, q.z), sphere radius q.w
  optixGetSphereData( gas, prim_idx, sbtGASIndex, 0.f, &q );
  float3 world_raypos = ray.origin + ray.tmax * ray.direction;
  float3 obj_raypos   = optixTransformPointFromWorldToObjectSpace( world_raypos );
  float3 N            = make_float3( (obj_raypos.x - q.x) / q.w, (obj_raypos.y - q.y) / q.w, (obj_raypos.z - q.z) / q.w );

  prd.sn = prd.gn = normalize((vec3f)N);
  
  vec3f tint = self.tint[prim_idx];
  vec3f ka = self.material.Ka * tint;
  vec3f kd = self.material.Kd * tint;
  vec3f ks = self.material.Ks;

  vec3f world_shading_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.sn));
  vec3f world_geometric_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.gn));
  vec3f ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  const vec2i launchIndex = owl::getLaunchIndex();
  const vec2i launchDims  = owl::getLaunchDims();
  // const bool dbg_x = launchIndex.x == launchDims.x/2;
  // const bool dbg_y = launchIndex.y == launchDims.y/2;
  // const bool dbg =  dbg_x & dbg_y;
  
  phongShade( kd, ka, ks, ffnormal, self.material.phong_exp, self.material.reflectivity );
}

OPTIX_ANY_HIT_PROGRAM(SpheresGeom)()
{
  PRD &prd = owl::getPRD<PRD>();
  prd.shadow.attenuation = 0.f;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();

  PRD &prd = owl::getPRD<PRD>();
  prd.radiance.result = self.bg_color;
}

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i launchIndex = owl::getLaunchIndex();
  // const bool dbg_x = launchIndex.x == self.fbSize.x/2;
  // const bool dbg_y = launchIndex.y == self.fbSize.y/2;
  // const bool dbg =  dbg_x & dbg_y;
  
  const auto &lp = optixLaunchParams;
  const int pixelID = launchIndex.x+self.fbSize.x*launchIndex.y; 

  // printf("pixel %i %i\n",launchIndex.x,launchIndex.y);
  
  Random rng(pixelID,lp.accumID);
  
  const vec2f screen = (vec2f(launchIndex)+vec2f(rng(),rng())) / vec2f(self.fbSize);
  RadianceRay ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);
  
  //ray.time = 0.5f;
  ray.time = 0.f;
  
  vec4f accumColor = 0.f;

  PRD prd;
  prd.t_hit = 1e20f;
  prd.radiance.importance = 1.f;

  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/prd,
                /*only CH*/OPTIX_RAY_FLAG_DISABLE_ANYHIT);

  accumColor += vec4f(prd.radiance.result,1.f);

  if (lp.accumID > 0)
    accumColor += vec4f(lp.accumBuffer[pixelID]);
  lp.accumBuffer[pixelID] = accumColor;
  accumColor *= (1.f/(lp.accumID+1));
  self.fbPtr[pixelID]
    = owl::make_rgba(vec3f(accumColor));
}

#endif
