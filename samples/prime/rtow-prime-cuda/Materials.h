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

#pragma once

#include <owl/owl_prime.h>

#include <cuda_runtime.h>
#include <random>
#include <owl/common/math/random.h>
#include <owl/common/math/AffineSpace.h>

namespace samples {

  using namespace owl;
  using namespace owl::common;

  struct Hit : public OPHit {};
  
  /*! C++ version of a "OPRay" - MUST match the memory layout of OPRay to work */
  struct Ray { vec3f origin; float tmin; vec3f direction; float tmax; };
  
  typedef owl::common::LCG<16> Random;

  inline __both__ float schlick(float cosine,
                                float ref_idx)
  {
    float r0 = (1.0f - ref_idx) / (1.0f + ref_idx);
    r0 = r0 * r0;
    return r0 + (1.0f - r0)*powf((1.0f - cosine), 5.0f);
  }

  inline __both__ bool refract(const vec3f &v,
                               const vec3f &n,
                               float ni_over_nt,
                               vec3f &refracted)
  {
    vec3f uv = normalize(v);
    float dt = dot(uv, n);
    float discriminant = 1.0f - ni_over_nt * ni_over_nt*(1 - dt * dt);
    if (discriminant > 0) {
      refracted = ni_over_nt * (uv - n * dt) - n * sqrtf(discriminant);
      return true;
    }
    else
      return false;
  }

  inline __both__  vec3f reflect(const vec3f &v, const vec3f &n)
  {
    return v - 2.0f*dot(v, n)*n;
  }

  inline __both__ vec3f random_in_unit_disk(Random &random) {
    vec3f p;
    do {
      p = 2.0f*vec3f(random(), random(), 0) - vec3f(1, 1, 0);
    } while (dot(p, p) >= 1.0f);
    return p;
  }


#define RANDVEC3F vec3f(rnd(),rnd(),rnd())

  inline __both__ vec3f random_in_unit_sphere(Random &rnd) {
    vec3f p;
    do {
      p = 2.0f*RANDVEC3F - vec3f(1.f, 1.f, 1.f);
    } while (dot(p,p) >= 1.0f);
    return p;
  }

  struct ScatterEvent {
    // in:
    vec3f inDir;
    vec3f P;
    vec3f N;
    // out
    vec3f out_dir;
    vec3f out_org;
    vec3f attenuation;
  };

  /*! host side code for the "Lambertian" material; the actual
    sampling code is in the programs/lambertian.cu closest hit program */
  struct Lambertian {
    /*! constructor */
    Lambertian(const vec3f &albedo) : albedo(albedo) {}
    
    /*! the actual scatter function - in Pete's reference code, that's
      a virtual function, but since we have a different function per
      program we do not need this here */
    inline __device__
    bool scatter(ScatterEvent &scatter,
                         Random   &random) 
    {
      if (dot(scatter.inDir,scatter.N) > 0.f)
        scatter.N = -scatter.N;
      vec3f target
        = scatter.P + (scatter.N + random_in_unit_sphere(random));
      
      // return scattering event
      scatter.out_org     = scatter.P;
      scatter.out_dir     = target-scatter.P;
      scatter.attenuation = albedo;
      return true;
    }

    
    const vec3f albedo;
  };

  /*! host side code for the "Metal" material; the actual
    sampling code is in the programs/metal.cu closest hit program */
  struct Metal {
    /*! constructor */
    Metal(const vec3f &albedo, const float fuzz) : albedo(albedo), fuzz(fuzz) {}
    /*! the actual scatter function - in Pete's reference code, that's a
      virtual function, but since we have a different function per program
      we do not need this here */
    inline __device__
    bool scatter(ScatterEvent &scatter,
                         Random   &random) 
    {
      if (dot(scatter.inDir,scatter.N) > 0.f)
        scatter.N = -scatter.N;
      vec3f reflected = reflect(normalize(scatter.inDir),scatter.N);
      scatter.out_org     = scatter.P;
      scatter.out_dir     = (reflected+fuzz*random_in_unit_sphere(random));
      scatter.attenuation = vec3f(1.f);
      return (dot(scatter.out_dir, scatter.N) > 0.f);
    }
    
    const vec3f albedo;
    const float fuzz;
  };

  /*! host side code for the "Dielectric" material; the actual
    sampling code is in the programs/dielectric.cu closest hit program */
  struct Dielectric {
    /*! constructor */
    Dielectric(const float ref_idx) : ref_idx(ref_idx) {}
    /*! the actual scatter function - in Pete's reference code, that's a
      virtual function, but since we have a different function per program
      we do not need this here */
    inline __device__
    bool scatter(ScatterEvent &scatter,
                 Random   &random) 
    {
      vec3f outward_normal;
      vec3f reflected = reflect(scatter.inDir, scatter.N);
      float ni_over_nt;
      scatter.attenuation = vec3f(1.f, 1.f, 1.f); 
      vec3f refracted;
      float reflect_prob;
      float cosine;

      float NdotD = dot(scatter.inDir, scatter.N);
      scatter.inDir = normalize(scatter.inDir);
      if (NdotD > 0.f) {
        outward_normal = -scatter.N;
        ni_over_nt = ref_idx;
        cosine = sqrtf(1.f - ref_idx*ref_idx*(1.f-NdotD*NdotD));
      }
      else {
        outward_normal = scatter.N;
        ni_over_nt = 1.f / ref_idx;
        cosine = -NdotD;
      }
      
      if (refract(scatter.inDir, outward_normal, ni_over_nt, refracted)) 
        reflect_prob = schlick(cosine, ref_idx);
      else 
        reflect_prob = 1.f;

      scatter.out_org = scatter.P;
      if (random() < reflect_prob) 
        scatter.out_dir = reflected;
      else 
        scatter.out_dir = refracted;
  
      return true;
    }

    const float ref_idx;
  };

  /*! abstraction for a material that can create, and parameterize,
    a newly created GI's material and closest hit program */
  struct Material {
    typedef enum { LAMBERTIAN, DIELECTRIC, METAL } Type;

    Material(const Lambertian &_lambertian)
      : type(LAMBERTIAN),
        lambertian(_lambertian)
    {}
    Material(const Dielectric &_dielectric)
      : type(DIELECTRIC),
        dielectric(_dielectric)
    {}
    Material(const Metal &_metal)
      : type(METAL),
        metal(_metal)
    {}

    inline __device__ bool scatter(ScatterEvent &scatter,
                                   Random   &random) {
      if (type == LAMBERTIAN)
        return lambertian.scatter(scatter,random);
      if (type == METAL)
        return metal.scatter(scatter,random);
      return dielectric.scatter(scatter,random);
    }
    union {
      Lambertian lambertian;
      Dielectric dielectric;
      Metal metal;
    };
    const Type type;
  };

}
