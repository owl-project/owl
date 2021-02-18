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

OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(self.camera.dir_00
                + screen.u * self.camera.dir_du
                + screen.v * self.camera.dir_dv);

  vec3f color;
  owl::traceRay(/*accel to trace against*/self.world,
                /*the ray to trace*/ray,
                /*prd*/color);
    
  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

OPTIX_CLOSEST_HIT_PROGRAM(TriangleMesh)()
{
  vec3f &prd = owl::getPRD<vec3f>();

  const TrianglesGeomData &self = owl::getProgramData<TrianglesGeomData>();
  
  // compute normal:
  const int   primID = optixGetPrimitiveIndex();
  const vec3i index  = self.index[primID];
  const vec3f &A     = self.vertex[index.x];
  const vec3f &B     = self.vertex[index.y];
  const vec3f &C     = self.vertex[index.z];
  const vec3f Ng     = normalize(cross(B-A,C-A));

  const unsigned int instanceID = optixGetInstanceId();
  const vec3f color = self.colorPerInstance[instanceID];

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*color;
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

inline __device__ float3 makeFaceNormal( float3 t0, float3 t1, float t)
{
    float3 N;
    if (t == t1.x) 
        //+X
        N = make_float3( 1.0f,  0.0f,  0.0f);
    else if (t == t0.x)
        //-X
        N = make_float3( -1.0f,  0.0f,  0.0f);
    else if (t == t1.y)
        //+Y
        N = make_float3( 0.0f,  1.0f,  0.0f);
    else if (t == t0.y)
        //-Y
        N = make_float3( 0.0f, -1.0f,  0.0f);
    else if (t == t1.z)
        //+Z,
        N = make_float3( 0.0f,  0.0f,  1.0f);
    else //if (t == t0.z)
        //-Z
        N = make_float3( 0.0f,  0.0f, -1.0f);

    return N;
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
  vec3f tnear = owl::min(t0, t1);
  vec3f tfar = owl::max(t0, t1);
  float tmin = reduce_max(tnear);
  float tmax = reduce_min(tfar);

  if (tmin <= tmax && tmin > ray_tmin && tmin < ray_tmax) {
    const float3 N = makeFaceNormal(t0, t1, tmin);
    optixReportIntersection( tmin, 0, float_as_int(N.x), float_as_int(N.y), float_as_int(N.z));
  }
}

OPTIX_CLOSEST_HIT_PROGRAM(VoxGeom)()
{
  // convert indices to 3d box
  const int primID = optixGetPrimitiveIndex();
  const VoxGeomData &self = owl::getProgramData<VoxGeomData>();

  vec3f &prd = owl::getPRD<vec3f>();

  // Select normal for whichever face we hit
  const float3 Nbox = make_float3(
        int_as_float(optixGetAttribute_0()),
        int_as_float(optixGetAttribute_1()),
        int_as_float(optixGetAttribute_2()));
  const vec3f Ng = normalize(vec3f(optixTransformNormalFromObjectToWorldSpace(Nbox)));

  // Convert 8 bit color to float
  const int ci = self.prims[primID].w;
  uchar4 col = self.colorPalette[ci];
  const vec3f color = vec3f(col.x, col.y, col.z) * (1.0f/255.0f);

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*color;


}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}

