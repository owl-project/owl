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

#include "owl-prime/deviceCode.h"
#include "owl-prime/Context.h"
#include "owl-prime/Triangles.h"

using namespace owl::common;
using op::Hit;

extern "C" __constant__ op::Context::LPData optixLaunchParams;

struct PRD {
  Hit hit;
  Hit lastHit;
};

  
/*! closest hit program: this fct gets called ONLY for regular (ie,
    NON-findfirst/findnext) traces, so the hit we see is the hit we
    store */
OPTIX_CLOSEST_HIT_PROGRAM(TrianglesCH)()
{
  const auto &self  = owl::getProgramData<op::Triangles::SBTData>();
  PRD &prd = owl::getPRD<PRD>();
  Hit &hit = prd.hit;
  hit.t = optixGetRayTmax();
  hit.primID = optixGetPrimitiveIndex();
  hit.geomDataValue = self.geomDataValue;
  hit.instID = optixGetInstanceIndex();
  hit.u      = optixGetTriangleBarycentrics().x;
  hit.v      = optixGetTriangleBarycentrics().y;
}

/*! any hit program: this fct gets called ONLY for find-fist or
     find-next traced, regular traces disable anyhit */
OPTIX_ANY_HIT_PROGRAM(TrianglesAH)()
{
  const auto &self  = owl::getProgramData<op::Triangles::SBTData>();
  PRD &prd = owl::getPRD<PRD>();
  float t = optixGetRayTmax();

  if (t > prd.lastHit.t)
    // ACCEPT this hit for the pipeline, and return, but do NOT store
    // it in 'our' hit
    return;

  int primID = optixGetPrimitiveIndex();
  int instID = optixGetInstanceIndex();
  uint64_t geomDataValue = self.geomDataValue;
  const Hit lastHit = prd.lastHit;
  Hit &thisHit = prd.hit;
  bool greaterThanLastHit
    =  (t > prd.lastHit.t)
    || (t == prd.lastHit.t
        && (instID > lastHit.instID
            || (instID == lastHit.instID
                && (primID > lastHit.primID
                    || (primID == lastHit.primID
                        && geomDataValue > lastHit.geomDataValue)))));
  bool lessThanCurrentHit
    =  (t > prd.lastHit.t)
    || (t == prd.lastHit.t
        && (instID > thisHit.instID
            || (instID == thisHit.instID
                && (primID > thisHit.primID
                    || (primID == thisHit.primID
                        && geomDataValue > thisHit.geomDataValue)))));
  
  if (greaterThanLastHit && lessThanCurrentHit) {
    thisHit.primID = primID;
    thisHit.instID = instID;
    thisHit.geomDataValue = self.geomDataValue;
    thisHit.u      = optixGetTriangleBarycentrics().x;
    thisHit.v      = optixGetTriangleBarycentrics().y;
  }
  optixIgnoreIntersection();
}

OPTIX_RAYGEN_PROGRAM(traceRays)()
{
  PRD prd;
  prd.hit.clear();
  int rayID
    = owl::getLaunchIndex().x
    + owl::getLaunchDims().x
    * owl::getLaunchIndex().y;

  auto &lp = optixLaunchParams;
  if (rayID >= lp.numRays) return;

  if (lp.activeIDs) {
    rayID = lp.activeIDs[rayID];
    if (rayID < 0) return;
  }

  owl::Ray ray = owl::Ray((const vec3f&)lp.rays[rayID].origin,
                          (const vec3f&)lp.rays[rayID].direction,
                          lp.rays[rayID].tMin,
                          lp.rays[rayID].tMax);
  if (ray.tmin < ray.tmax) {
    uint32_t rayFlags = 0;
    if (lp.flags & OP_TRACE_FIND_FIRST) {
      // we need an anyhit program
      rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
      prd.lastHit.clear();
      prd.lastHit.t = ray.tmin;
    } else if (lp.flags & OP_TRACE_FIND_NEXT) {
      // we need an anyhit program, *and* need to know the last hit
      prd.lastHit = lp.hits[rayID];
      ray.tmin = nextafter(prd.lastHit.t,-1.f);
      rayFlags = OPTIX_RAY_FLAG_DISABLE_CLOSESTHIT;
    } else {
      prd.lastHit.clear();
      rayFlags = OPTIX_RAY_FLAG_DISABLE_ANYHIT;
    }
    owl::traceRay(lp.model,ray,prd,rayFlags);
  }
  lp.hits[rayID] = prd.hit;
}

