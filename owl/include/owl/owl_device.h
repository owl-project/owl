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

#pragma once

#include "owl/common/math/vec.h"
#include "owl/common/math/box.h"
#include "owl/common/math/AffineSpace.h"
// the 'actual' optix
#include <cuda.h>
#include <optix.h>
#include <vector_types.h>

// ==================================================================
// actual device-side "API" built-ins.
// ==================================================================

#ifndef __CUDACC__
#  error "this file should only ever get included on the device side"
#endif

namespace owl {

  using namespace owl::common;

  inline __device__ vec2i getLaunchIndex()
  {
    return (vec2i)optixGetLaunchIndex();
  }

  /*! return dimensions of a 2-dimensional optix launch. For 1- or
    3-dimensional launches we'll need separate functions */
  inline __device__ vec2i getLaunchDims()
  {
    return (vec2i)optixGetLaunchDimensions();
  }

  /*! return pointer to currently running program's "SBT Data" (which
      is pretty much what in owl we call the Program Data/Program
      Variables Struct. This method returns an untyped pointer, for
      automatic type conversion see the getProgramData<T> template */
  inline __device__ const void *getProgramDataPointer()
  {
    return (const void*)optixGetSbtDataPointer();
  }

  /*! convenience type-tagged version of \see
      getProgramDataPointer. Note this function does _not_ perform any
      type-checks, it's just hard-casting the SBT pointer to the
      expected type. */
  template<typename T>
  inline __device__ const T &getProgramData()
  {
    return *(const T*)getProgramDataPointer();
  }


  // ==================================================================
  // general convenience/helper functions - may move to samples
  // ==================================================================
  inline __device__ float linear_to_srgb(float x) {
    if (x <= 0.0031308f) {
      return 12.92f * x;
    }
    return 1.055f * pow(x, 1.f/2.4f) - 0.055f;
  }

  inline __device__ uint32_t make_8bit(const float f)
  {
    return min(255,max(0,int(f*256.f)));
  }

  inline __device__ uint32_t make_rgba(const vec3f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (0xffU << 24);
  }
  inline __device__ uint32_t make_rgba(const vec4f color)
  {
    return
      (make_8bit(color.x) << 0) +
      (make_8bit(color.y) << 8) +
      (make_8bit(color.z) << 16) +
      (make_8bit(color.w) << 24);
  }

  inline __device__ void initializeTransformToIdentity(float (&t)[12]) {
    t[0] = 1.f; t[1] = 0.f; t[2] = 0.f; t[3] = 0.f;
    t[4] = 0.f; t[5] = 1.f; t[6] = 0.f; t[7] = 0.f;
    t[8] = 0.f; t[9] = 0.f; t[10] = 1.f; t[11] = 0.f;
  }

  inline __device__ void toRowMajor(affine3f tfm, float (&t)[12]) {
    t[0] = tfm.l.vx.x; t[1] = tfm.l.vx.y; t[2] = tfm.l.vx.z; t[3] = tfm.p.x;
    t[4] = tfm.l.vy.x; t[5] = tfm.l.vy.y; t[6] = tfm.l.vy.z; t[7] = tfm.p.y;
    t[8] = tfm.l.vz.x; t[9] = tfm.l.vz.y; t[10] = tfm.l.vz.z; t[11] = tfm.p.z;
  }

  static __forceinline__ __device__ void* unpackPointer( uint32_t i0, uint32_t i1 )
  {
    const uint64_t uptr = static_cast<uint64_t>( i0 ) << 32 | i1;
    void*           ptr = reinterpret_cast<void*>( uptr ); 
    return ptr;
  }


  static __forceinline__ __device__ void  packPointer( void* ptr, uint32_t& i0, uint32_t& i1 )
  {
    const uint64_t uptr = reinterpret_cast<uint64_t>( ptr );
    i0 = uptr >> 32;
    i1 = uptr & 0x00000000ffffffff;
  }


  static __forceinline__ __device__ void *getPRDPointer()
  { 
    const uint32_t u0 = optixGetPayload_0();
    const uint32_t u1 = optixGetPayload_1();
    return unpackPointer(u0, u1);
  }

  template<typename T>
  static __forceinline__ __device__ T &getPRD()
  { return *(T*)getPRDPointer(); }

  template<int _rayType=0, int _numRayTypes=1, bool _disablePerGeometrySBTRecords=0>
  struct RayT {
    enum { rayType = _rayType };
    enum { numRayTypes = _numRayTypes };
    enum { disablePerGeometrySBTRecords = _disablePerGeometrySBTRecords };
    inline __device__ RayT() {}
    inline __device__ RayT(const vec3f &origin,
                          const vec3f &direction,
                          float tmin,
                          float tmax,
                          OptixVisibilityMask visibilityMask=(OptixVisibilityMask)(-1))
      : origin(origin),
        direction(direction),
        tmin(tmin),
        tmax(tmax),
        visibilityMask(visibilityMask)
    {}
    
    vec3f origin, direction;
    float tmin=0.f,tmax=1e30f,time=0.f;
    OptixVisibilityMask visibilityMask=(OptixVisibilityMask)-1;
  };
  typedef RayT<0,1> Ray;


  template<typename RayType, typename PRD>
  inline __device__
  void traceRay(OptixTraversableHandle traversable,
                const RayType &ray,
                PRD           &prd,
                uint32_t rayFlags = 0u)
  {
    unsigned int           p0 = 0;
    unsigned int           p1 = 0;
    owl::packPointer(&prd,p0,p1);

    optixTrace(traversable,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               ray.time,
               ray.visibilityMask,
               /*rayFlags     */ rayFlags,
               /*SBToffset    */ ray.rayType,
               /*SBTstride    */ ray.numRayTypes * (ray.disablePerGeometrySBTRecords) ? 0 : 1,
               /*missSBTIndex */ ray.rayType,              
               p0,
               p1);
  }

  template<typename PRD>
  inline __device__
  void trace(OptixTraversableHandle traversable,
             const Ray &ray,
             int numRayTypes,
             PRD &prd,
             int sbtOffset = 0)
  {
    unsigned int           p0 = 0;
    unsigned int           p1 = 0;
    owl::packPointer(&prd,p0,p1);
    
    optixTrace(traversable,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               ray.time,
               ray.visibilityMask,
               /*rayFlags     */0u,
               /*SBToffset    */ray.rayType + numRayTypes*sbtOffset,
               /*SBTstride    */numRayTypes,
               /*missSBTIndex */ray.rayType,
               p0,
               p1);
  }

  template<typename PRD>
  inline __device__
  void trace(OptixTraversableHandle traversable,
             const Ray &ray,
             int numRayTypes,
             bool disablePerGeometrySBTRecords,
             PRD &prd,
             int sbtOffset = 0)
  {
    unsigned int           p0 = 0;
    unsigned int           p1 = 0;
    owl::packPointer(&prd,p0,p1);
    
    optixTrace(traversable,
               (const float3&)ray.origin,
               (const float3&)ray.direction,
               ray.tmin,
               ray.tmax,
               ray.time,
               ray.visibilityMask,
               /*rayFlags     */0u,
               /*SBToffset    */ray.rayType + numRayTypes*sbtOffset,
               /*SBTstride    */numRayTypes * (ray.disablePerGeometrySBTRecords) ? 0 : 1,
               /*missSBTIndex */ray.rayType,
               p0,
               p1);
  }
  
} // ::owl

/*! in owl we can only change the _type_ of launch params, they always
    need to be caleld 'optixLaunchParams', and must have __constant__
    storage*/
#define DECLARE_OPTIX_LAUNCH_PARAMS(type) extern __constant__ type optixLaunchParams;

#define OPTIX_RAYGEN_PROGRAM(programName) \
  extern "C" __global__ \
  void __raygen__##programName

#define OPTIX_CLOSEST_HIT_PROGRAM(programName) \
  extern "C" __global__ \
  void __closesthit__##programName

#define OPTIX_ANY_HIT_PROGRAM(programName) \
  extern "C" __global__ \
  void __anyhit__##programName

#define OPTIX_INTERSECT_PROGRAM(programName) \
  extern "C" __global__ \
  void __intersection__##programName

#define OPTIX_MISS_PROGRAM(programName) \
  extern "C" __global__ \
  void __miss__##programName

/* defines the wrapper stuff to actually launch all the bounds
   programs from the host - todo: move to deviceAPI.h once working */
#ifndef OPTIX_BOUNDS_PROGRAM
#define OPTIX_BOUNDS_PROGRAM(progName)                                  \
  /* fwd decl for the kernel func to call */                            \
  inline __device__                                                     \
  void __boundsFunc__##progName(const void *geomData,                   \
                                owl::common::box3f &bounds,             \
                                const int32_t primID);                  \
                                                                        \
  /* the '__global__' kernel we can get a function handle on */         \
  extern "C" __global__                                                 \
  void __boundsFuncKernel__##progName(const void  *geomData,            \
                         owl::common::box3f *const boundsArray,         \
                                      const uint32_t numPrims)          \
  {                                                                     \
    uint32_t blockIndex                                                 \
      = blockIdx.x                                                      \
      + blockIdx.y * gridDim.x                                          \
      + blockIdx.z * gridDim.x * gridDim.y;                             \
    uint32_t primID                                                     \
      = threadIdx.x + blockDim.x*threadIdx.y                            \
      + blockDim.x*blockDim.y*blockIndex;                               \
    if (primID < numPrims) {                                            \
      __boundsFunc__##progName(geomData,boundsArray[primID],primID);    \
    }                                                                   \
  }                                                                     \
                                                                        \
  /* now the actual device code that the user is writing: */            \
  inline __device__ void __boundsFunc__##progName                       \
  /* program args and body supplied by user ... */
#endif

/* defines the wrapper stuff to actually launch all the bounds
   programs from the host - todo: move to deviceAPI.h once working */
#ifndef OPTIX_MOTION_BOUNDS_PROGRAM
#define OPTIX_MOTION_BOUNDS_PROGRAM(progName)                                  \
  /* fwd decl for the kernel func to call */                            \
  inline __device__                                                     \
  void __motionBoundsFunc__##progName(const void *geomData,                   \
                                owl::common::box3f &boundskey1,         \
                                owl::common::box3f &boundskey2,         \
                                const int32_t primID);                  \
                                                                        \
  /* the '__global__' kernel we can get a function handle on */         \
  extern "C" __global__                                                 \
  void __motionBoundsFuncKernel__##progName(const void  *geomData,      \
                         owl::common::box3f *const boundsArrayKey1,     \
                         owl::common::box3f *const boundsArrayKey2,     \
                                      const uint32_t numPrims)          \
  {                                                                     \
    uint32_t blockIndex                                                 \
      = blockIdx.x                                                      \
      + blockIdx.y * gridDim.x                                          \
      + blockIdx.z * gridDim.x * gridDim.y;                             \
    uint32_t primID                                                     \
      = threadIdx.x + blockDim.x*threadIdx.y                            \
      + blockDim.x*blockDim.y*blockIndex;                               \
    if (primID < numPrims) {                                            \
      __motionBoundsFunc__##progName(geomData,                          \
                                     boundsArrayKey1[primID],           \
                                     boundsArrayKey2[primID],           \
                                     primID);                           \
    }                                                                   \
  }                                                                     \
                                                                        \
  /* now the actual device code that the user is writing: */            \
  inline __device__ void __motionBoundsFunc__##progName                 \
  /* program args and body supplied by user ... */
#endif

/* defines a wrapper to a new program type which enables GPU-side 
 instance manipulation (transforms, visibility mask, etc). Assumes 
 that geometry contribution to hitgroup index is disabled. */
#ifndef OPTIX_INSTANCE_PROGRAM
#define OPTIX_INSTANCE_PROGRAM(progName)                                \
  /* fwd decl for the kernel func to call */                            \
  inline __device__                                                     \
  void __instanceFunc__##progName(                                      \
    const int32_t instanceIndex, OptixInstance &instance);              \
                                                                        \
  /* the '__global__' kernel we can get a function handle on */         \
  extern "C" __global__                                                 \
  void __instanceFuncKernel__##progName(                                \
    OptixInstance *insts, uint32_t numInsts, uint32_t numRayTypes)      \
  {                                                                     \
    uint32_t blockIndex                                                 \
      = blockIdx.x                                                      \
      + blockIdx.y * gridDim.x                                          \
      + blockIdx.z * gridDim.x * gridDim.y;                             \
    uint32_t instanceIndex                                              \
      = threadIdx.x + blockDim.x*threadIdx.y                            \
      + blockDim.x*blockDim.y*blockIndex;                               \
    if (instanceIndex < numInsts) {                                     \
      OptixInstance oi    = {};                                         \
      /* defaults */                                                    \
      oi.flags             = OPTIX_INSTANCE_FLAG_NONE;                  \
      oi.instanceId        = instanceIndex;                             \
      oi.visibilityMask = 255;                                          \
      oi.traversableHandle = 0; /* if not set, ignored by builder */    \
      initializeTransformToIdentity(oi.transform);                      \
      __instanceFunc__##progName(instanceIndex, oi);                    \
      insts[instanceIndex] = oi;                                        \
    }                                                                   \
  }                                                                     \
                                                                        \
  /* now the actual device code that the user is writing: */            \
  inline __device__ void __instanceFunc__##progName                     \
  /* program args and body supplied by user ... */
#endif
