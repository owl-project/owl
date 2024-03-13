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

/* this sample uses features from newer versions of optix; it will not
   run with older versions, nor even compile with them - so to make
   the entire project compile with older versions of optix we'll here
   disable this newer code if an older version is being used. */
#if OPTIX_VERSION >= 70300

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

/*! stolen from optixHair sample in OptiX 7.4 SDK */
// Get curve hit-point in world coordinates.
static __forceinline__ __device__ vec3f getHitPoint()
{
  const float  t            = optixGetRayTmax();
  const float3 rayOrigin    = optixGetWorldRayOrigin();
  const float3 rayDirection = optixGetWorldRayDirection();

  return (vec3f)rayOrigin + t * (vec3f)rayDirection;
}



/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
//
// First order polynomial interpolator
//
struct LinearBSplineSegment
{
  __device__ __forceinline__ LinearBSplineSegment() {}
  __device__ __forceinline__ LinearBSplineSegment( const vec4f* q ) { initialize( q ); }

  __device__ __forceinline__ void initialize( const vec4f* q )
  {
    p[0] = q[0];
    p[1] = q[1] - q[0];  // pre-transform p[] for fast evaluation
  }

  __device__ __forceinline__ float radius( const float& u ) const { return p[0].w + p[1].w * u; }

  __device__ __forceinline__ vec3f position3( float u ) const { return (vec3f&)p[0] + u * (vec3f&)p[1]; }
  __device__ __forceinline__ vec4f position4( float u ) const { return p[0] + u * p[1]; }

  __device__ __forceinline__ float min_radius( float u1, float u2 ) const
  {
    return fminf( radius( u1 ), radius( u2 ) );
  }

  __device__ __forceinline__ float max_radius( float u1, float u2 ) const
  {
    if( !p[1].w )
      return p[0].w;  // a quick bypass for constant width
    return fmaxf( radius( u1 ), radius( u2 ) );
  }

  __device__ __forceinline__ vec3f velocity3( float u ) const { return (vec3f&)p[1]; }
  __device__ __forceinline__ vec4f velocity4( float u ) const { return p[1]; }

  __device__ __forceinline__ vec3f acceleration3( float u ) const { return vec3f( 0.f ); }
  __device__ __forceinline__ vec4f acceleration4( float u ) const { return vec4f( 0.f ); }

  __device__ __forceinline__ float derivative_of_radius( float u ) const { return p[1].w; }

  vec4f p[2];  // pre-transformed "control points" for fast evaluation
};


/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
//
// Second order polynomial interpolator
//
struct QuadraticBSplineSegment
{
  __device__ __forceinline__ QuadraticBSplineSegment() {}
  __device__ __forceinline__ QuadraticBSplineSegment( const vec4f* q ) { initializeFromBSpline( q ); }

  __device__ __forceinline__ void initializeFromBSpline( const vec4f* q )
  {
    // pre-transform control-points for fast evaluation
    p[0] = q[1] / 2.0f + q[0] / 2.0f;
    p[1] = q[1] - q[0];
    p[2] = q[0] / 2.0f - q[1] + q[2] / 2.0f;
  }

  __device__ __forceinline__ void export2BSpline( vec4f bs[3] ) const
  {
    bs[0] = p[0] - p[1] / 2.f;
    bs[1] = p[0] + p[1] / 2.f;
    bs[2] = p[0] + 1.5f * p[1] + 2.f * p[2];
  }

  __device__ __forceinline__ vec3f position3( float u ) const
  {
    return (vec3f&)p[0] + u * (vec3f&)p[1] + u * u * (vec3f&)p[2];
  }
  __device__ __forceinline__ vec4f position4( float u ) const { return p[0] + u * p[1] + u * u * p[2]; }

  __device__ __forceinline__ float radius( float u ) const { return p[0].w + u * ( p[1].w + u * p[2].w ); }

  __device__ __forceinline__ float min_radius( float u1, float u2 ) const
  {
    float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
    return fminf( fminf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
  }

  __device__ __forceinline__ float max_radius( float u1, float u2 ) const
  {
    if( !p[1].w && !p[2].w )
      return p[0].w;  // a quick bypass for constant width
    float root1 = clamp( -0.5f * p[1].w / p[2].w, u1, u2 );
    return fmaxf( fmaxf( radius( u1 ), radius( u2 ) ), radius( root1 ) );
  }

  __device__ __forceinline__ vec3f velocity3( float u ) const { return (vec3f&)p[1] + 2.f * u * (vec3f&)p[2]; }
  __device__ __forceinline__ vec4f velocity4( float u ) const { return p[1] + 2.f * u * p[2]; }

  __device__ __forceinline__ vec3f acceleration3( float u ) const { return 2.f * (vec3f&)p[2]; }
  __device__ __forceinline__ vec4f acceleration4( float u ) const { return 2.f * p[2]; }

  __device__ __forceinline__ float derivative_of_radius( float u ) const { return p[1].w + 2.f * u * p[2].w; }

  vec4f p[3];  // pre-transformed "control points" for fast evaluation
};

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
//
// Third order polynomial interpolator
//
struct CubicBSplineSegment
{
  __device__ __forceinline__ CubicBSplineSegment() {}
  __device__ __forceinline__ CubicBSplineSegment( const vec4f* q ) { initializeFromBSpline( q ); }

  __device__ __forceinline__ void initializeFromBSpline( const vec4f* q )
  {
    // pre-transform control points for fast evaluation
    p[0] = ( q[2] + q[0] ) / 6.f + ( 4.f / 6.f ) * q[1];
    p[1] = q[2] - q[0];
    p[2] = q[2] - q[1];
    p[3] = q[3] - q[1];
  }

  __device__ __forceinline__ void export2BSpline( vec4f bs[4] ) const
  {
    // inverse of initializeFromBSpline
    bs[0] = p[0] + ( 4.f * p[2] - 5.f * p[1] ) / 6.f;
    bs[1] = p[0] + ( p[1] - 2.f * p[2] ) / 6.f;
    bs[2] = p[0] + ( p[1] + 4.f * p[2] ) / 6.f;
    bs[3] = p[0] + p[3] + ( p[1] - 2.f * p[2] ) / 6.f;
  }

  __device__ __forceinline__ static vec3f terms( float u )
  {
    float uu = u * u;
    float u3 = ( 1 / 6.0f ) * uu * u;
    return vec3f( u3 + 0.5f * ( u - uu ), uu - 4.f * u3, u3 );
  }

  __device__ __forceinline__ vec3f position3( float u ) const
  {
    vec3f q = terms( u );
    return (vec3f&)p[0] + q.x * (vec3f&)p[1] + q.y * (vec3f&)p[2] + q.z * (vec3f&)p[3];
  }
  __device__ __forceinline__ vec4f position4( float u ) const
  {
    vec3f q = terms( u );
    return p[0] + q.x * p[1] + q.y * p[2] + q.z * p[3];
  }

  __device__ __forceinline__ float radius( float u ) const
  {
    return p[0].w + u * ( p[1].w / 2 + u * ( ( p[2].w - p[1].w / 2 ) + u * ( p[1].w - 4 * p[2].w + p[3].w ) / 6 ) );
  }

  __device__ __forceinline__ float min_radius( float u1, float u2 ) const
  {
    // a + 2 b u - c u^2
    float a    = p[1].w;
    float b    = 2 * p[2].w - p[1].w;
    float c    = 4 * p[2].w - p[1].w - p[3].w;
    float rmin = fminf( radius( u1 ), radius( u2 ) );
    if( fabsf( c ) < 1e-5f )
      {
        float root1 = clamp( -0.5f * a / b, u1, u2 );
        return fminf( rmin, radius( root1 ) );
      }
    else
      {
        float det   = b * b + a * c;
        det         = det <= 0.0f ? 0.0f : sqrtf( det );
        float root1 = clamp( ( b + det ) / c, u1, u2 );
        float root2 = clamp( ( b - det ) / c, u1, u2 );
        return fminf( rmin, fminf( radius( root1 ), radius( root2 ) ) );
      }
  }

  __device__ __forceinline__ float max_radius( float u1, float u2 ) const
  {
    if( !p[1].w && !p[2].w && !p[3].w )
      return p[0].w;  // a quick bypass for constant width
    // a + 2 b u - c u^2
    float a    = p[1].w;
    float b    = 2 * p[2].w - p[1].w;
    float c    = 4 * p[2].w - p[1].w - p[3].w;
    float rmax = fmaxf( radius( u1 ), radius( u2 ) );
    if( fabsf( c ) < 1e-5f )
      {
        float root1 = clamp( -0.5f * a / b, u1, u2 );
        return fmaxf( rmax, radius( root1 ) );
      }
    else
      {
        float det   = b * b + a * c;
        det         = det <= 0.0f ? 0.0f : sqrtf( det );
        float root1 = clamp( ( b + det ) / c, u1, u2 );
        float root2 = clamp( ( b - det ) / c, u1, u2 );
        return fmaxf( rmax, fmaxf( radius( root1 ), radius( root2 ) ) );
      }
  }

  __device__ __forceinline__ vec3f velocity3( float u ) const
  {
    // adjust u to avoid problems with tripple knots.
    if( u == 0 )
      u = 0.000001f;
    if( u == 1 )
      u = 0.999999f;
    float v = 1 - u;
    return 0.5f * v * v * (vec3f&)p[1] + 2 * v * u * (vec3f&)p[2] + 0.5f * u * u * (vec3f&)p[3];
  }

  __device__ __forceinline__ vec4f velocity4( float u ) const
  {
    // adjust u to avoid problems with tripple knots.
    if( u == 0 )
      u = 0.000001f;
    if( u == 1 )
      u = 0.999999f;
    float v = 1 - u;
    return 0.5f * v * v * p[1] + 2 * v * u * p[2] + 0.5f * u * u * p[3];
  }

  __device__ __forceinline__ vec3f acceleration3( float u ) const { return vec3f( acceleration4( u ) ); }
  __device__ __forceinline__ vec4f acceleration4( float u ) const
  {
    return 2.f * p[2] - p[1] + ( p[1] - 4.f * p[2] + p[3] ) * u;
  }

  __device__ __forceinline__ float derivative_of_radius( float u ) const
  {
    float v = 1 - u;
    return 0.5f * v * v * p[1].w + 2 * v * u * p[2].w + 0.5f * u * u * p[3].w;
  }

  vec4f p[4];  // pre-transformed "control points" for fast evaluation
};

// Compute curve primitive surface normal in object space.
//
// Template parameters:
//   CurveType - A B-Spline evaluator class.
//   type - 0     ~ cylindrical approximation (correct if radius' == 0)
//          1     ~ conic       approximation (correct if curve'' == 0)
//          other ~ the bona fide surface normal
//
// Parameters:
//   bc - A B-Spline evaluator object.
//   u  - segment parameter of hit-point.
//   ps - hit-point on curve's surface in object space; usually
//        computed like this.
//        float3 ps = ray_orig + t_hit * ray_dir;
//        the resulting point is slightly offset away from the
//        surface. For this reason (Warning!) ps gets modified by this
//        method, projecting it onto the surface
//        in case it is not already on it. (See also inline
//        comments.)
//
template <typename CurveType, int type = 2>
__device__ __forceinline__ vec3f surfaceNormal( const CurveType& bc, float u, vec3f& ps )
{
  vec3f normal;
  if( u == 0.0f )
    {
      normal = -bc.velocity3( 0 );  // special handling for flat endcaps
    }
  else if( u == 1.0f )
    {
      normal = bc.velocity3( 1 );   // special handling for flat endcaps
    }
  else
    {
      // ps is a point that is near the curve's offset surface,
      // usually ray.origin + ray.direction * rayt.
      // We will push it exactly to the surface by projecting it to the plane(p,d).
      // The function derivation:
      // we (implicitly) transform the curve into coordinate system
      // {p, o1 = normalize(ps - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
      // curve'(t) = (0, length(d), 0); ps = (r, 0, 0);
      vec4f p4 = bc.position4( u );
      vec3f p  = vec3f( p4 );
      float  r  = p4.w;  // == length(ps - p) if ps is already on the surface
      vec4f d4 = bc.velocity4( u );
      vec3f d  = vec3f( d4 );
      float  dr = d4.w;
      float  dd = dot( d, d );

      vec3f o1 = ps - p;               // dot(modified_o1, d) == 0 by design:
      o1 -= ( dot( o1, d ) / dd ) * d;  // first, project ps to the plane(p,d)
      o1 *= r / length( o1 );           // and then drop it to the surface
      ps = p + o1;                      // fine-tuning the hit point
      if( type == 0 )
        {
          normal = o1;  // cylindrical approximation
        }
      else
        {
          if( type != 1 )
            {
              dd -= dot( bc.acceleration3( u ), o1 );
            }
          normal = dd * o1 - ( dr * r ) * d;
        }
    }
  return normalize( normal );
}

template <int type = 1>
__device__ __forceinline__ vec3f surfaceNormal( const LinearBSplineSegment& bc, float u, vec3f& ps )
{
  vec3f normal;
  if( u == 0.0f )
    {
      normal = ps - (vec3f&)(bc.p[0]);  // special handling for round endcaps
    }
  else if( u >= 1.0f )
    {
      // reconstruct second control point (Note: the interpolator pre-transforms
      // the control-points to speed up repeated evaluation.
      const vec3f p1 = (vec3f&)(bc.p[1]) + (vec3f&)(bc.p[0]);
      normal = ps - p1;  // special handling for round endcaps
    }
  else
    {
      // ps is a point that is near the curve's offset surface,
      // usually ray.origin + ray.direction * rayt.
      // We will push it exactly to the surface by projecting it to the plane(p,d).
      // The function derivation:
      // we (implicitly) transform the curve into coordinate system
      // {p, o1 = normalize(ps - p), o2 = normalize(curve'(t)), o3 = o1 x o2} in which
      // curve'(t) = (0, length(d), 0); ps = (r, 0, 0);
      vec4f p4 = bc.position4( u );
      vec3f p  = vec3f( p4 );
      float  r  = p4.w;  // == length(ps - p) if ps is already on the surface
      vec4f d4 = bc.velocity4( u );
      vec3f d  = vec3f( d4 );
      float  dr = d4.w;
      float  dd = dot( d, d );

      vec3f o1 = ps - p;               // dot(modified_o1, d) == 0 by design:
      o1 -= ( dot( o1, d ) / dd ) * d;  // first, project ps to the plane(p,d)
      o1 *= r / length( o1 );           // and then drop it to the surface
      ps = p + o1;                      // fine-tuning the hit point
      if( type == 0 )
        {
          normal = o1;  // cylindrical approximation
        }
      else
        {
          normal = dd * o1 - ( dr * r ) * d;
        }
    }
  return normalize( normal );
}

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ vec3f normalLinear( const int primitiveIndex )
{
  const OptixTraversableHandle gas = optixGetGASTraversableHandle();
  const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
  vec4f                       controlPoints[2];

  optixGetLinearCurveVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, (float4*)controlPoints );

  LinearBSplineSegment interpolator( controlPoints );
  vec3f               hitPoint = getHitPoint();
  // interpolators work in object space
  hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
  const vec3f normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
  return optixTransformNormalFromObjectToWorldSpace( normal );
}

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
// Compute surface normal of quadratic pimitive in world space.
static __forceinline__ __device__ vec3f normalQuadratic( const int primitiveIndex )
{
  const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
  const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
  vec4f                       controlPoints[3];

  optixGetQuadraticBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, (float4*)controlPoints );

  QuadraticBSplineSegment interpolator( controlPoints );
  vec3f                  hitPoint = getHitPoint();
  // interpolators work in object space
  hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
  const vec3f normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
  return optixTransformNormalFromObjectToWorldSpace( normal );
}

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
// Compute surface normal of cubic pimitive in world space.
static __forceinline__ __device__ vec3f normalCubic( const int primitiveIndex )
{
  const OptixTraversableHandle gas         = optixGetGASTraversableHandle();
  const unsigned int           gasSbtIndex = optixGetSbtGASIndex();
  vec4f                       controlPoints[4];

  optixGetCubicBSplineVertexData( gas, primitiveIndex, gasSbtIndex, 0.0f, (float4*)controlPoints );

  CubicBSplineSegment interpolator( controlPoints );
  vec3f              hitPoint = getHitPoint();
  // interpolators work in object space
  hitPoint            = optixTransformPointFromWorldToObjectSpace( hitPoint );
  const vec3f normal = surfaceNormal( interpolator, optixGetCurveParameter(), hitPoint );
  return optixTransformNormalFromObjectToWorldSpace( normal );
}

/*! compute normal - stolen from optixHair sample in OptiX 7.4 SDK */
// Compute normal
//
static __forceinline__ __device__ vec3f computeNormal( OptixPrimitiveType type, const int primitiveIndex )
{
  switch( type ) {
  case OPTIX_PRIMITIVE_TYPE_ROUND_LINEAR:
    return  normalLinear( primitiveIndex );
    break;
  case OPTIX_PRIMITIVE_TYPE_ROUND_QUADRATIC_BSPLINE:
    return normalQuadratic( primitiveIndex );
    break;
  case OPTIX_PRIMITIVE_TYPE_ROUND_CUBIC_BSPLINE:
    return  normalCubic( primitiveIndex );
    break;
  }
  return vec3f(0.0f);
}

static
__device__ void phongShade( vec3f p_Kd,
                            vec3f p_Ka,
                            vec3f p_Ks,
                            vec3f p_normal,
                            float p_phong_exp,
                            vec3f p_reflectivity )
{
  const auto &self
    = owl::getProgramData<CurvesGeomData>();

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

      result += p_Kd * nDl * Lc * (vec3f(1.f,1.f,1.f)-p_reflectivity);

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
    new_prd.max_depth=prd.max_depth;

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

OPTIX_CLOSEST_HIT_PROGRAM(CurvesGeom)()
{
  const auto &self
    = owl::getProgramData<CurvesGeomData>();

  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.origin = optixGetWorldRayOrigin();
  ray.direction = optixGetWorldRayDirection();
  ray.tmin = optixGetRayTmin();
  ray.tmax = optixGetRayTmax();

  prd.t_hit = optixGetRayTmax(); // TODO:
  prd.sn = prd.gn
    // = vec3f(0,1,0)
    = computeNormal(optixGetPrimitiveType(), optixGetPrimitiveIndex());


  // Get curve material properties
  const auto &curveMaterial = owl::getProgramData<CurvesGeom>().curves[optixGetPrimitiveIndex()];
  Material material = curveMaterial.material;
  vec3f ka = material.Ka;
  vec3f kd = material.Kd;
  vec3f ks = material.Ks;

  vec3f world_shading_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.sn));
  vec3f world_geometric_normal = normalize((vec3f)optixTransformNormalFromObjectToWorldSpace(prd.gn));
  vec3f ffnormal  = faceforward( world_shading_normal, -ray.direction, world_geometric_normal );

  const vec2i launchIndex = owl::getLaunchIndex();
  const vec2i launchDims  = owl::getLaunchDims();
  // const bool dbg_x = launchIndex.x == launchDims.x/2;
  // const bool dbg_y = launchIndex.y == launchDims.y/2;
  // const bool dbg =  dbg_x & dbg_y;
  
  phongShade( kd, ka, ks, ffnormal, material.phong_exp, material.reflectivity );
}

OPTIX_ANY_HIT_PROGRAM(CurvesGeom)()
{
  PRD &prd = owl::getPRD<PRD>();
  prd.shadow.attenuation = 0.f;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const MissProgData &self = owl::getProgramData<MissProgData>();
  PRD &prd = owl::getPRD<PRD>();

  RadianceRay ray;
  ray.direction = optixGetWorldRayDirection();

  const vec3f rayDir = normalize(ray.direction);
  const float t = 0.5f*(rayDir.z + 1.0f);
  prd.radiance.result = (1.0f - t) * vec3f(0.8f,0.71f,0.71f) + t * self.bg_color;
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
#if 0
  if (dbg) printf("camera DOF %f %f\n",self.camera.focal_scale,self.camera.aperture_radius);
  vec3f ray_target = ray.origin + self.camera.focal_scale * ray.direction;
  // lens sampling
  vec2f sample = square_to_disk(make_float2(rng(), rng()));
  ray.origin = ray.origin + self.camera.aperture_radius * ( sample.x * normalize( self.camera.dir_du ) +  sample.y * normalize( self.camera.dir_dv ) );
  ray.direction = normalize(ray_target - ray.origin);
#endif
  
  //ray.time = 0.5f;
  ray.time = 0.f;
  
  vec4f accumColor = 0.f;

  PRD prd;
  prd.t_hit = 1e20f;
  prd.radiance.importance = 1.f;
  prd.max_depth = 5;

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
