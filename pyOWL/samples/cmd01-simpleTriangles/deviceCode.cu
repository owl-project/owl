
// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include <owl/pyOWL.h>
#include <owl/owl.h>
#include <owl/common/math/vec.h>
#include <optix_device.h>


using namespace owl;

/* variables for the triangle mesh geometry */
struct TrianglesGeomData
{
  /*! base color we use for the entire mesh */
  vec3f color;
  /*! array/buffer of vertex indices */
  vec3i *index;
  /*! array/buffer of vertex positions */
  vec3f *vertex;
};

/* variables for the ray generation program */
struct RayGenData
{
  uint32_t *fbPtr;
  vec2i  fbSize;
  OptixTraversableHandle world;

  vec3f camera_pos;
  vec3f camera_dir_00;
  vec3f camera_dir_du;
  vec3f camera_dir_dv;
};

/* variables for the miss program */
struct MissProgData
{
  vec3f  color0;
  vec3f  color1;
};



OPTIX_RAYGEN_PROGRAM(simpleRayGen)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera_pos;
  ray.direction 
    = normalize(self.camera_dir_00
                + screen.u * self.camera_dir_du
                + screen.v * self.camera_dir_dv);

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

  const vec3f rayDir = optixGetWorldRayDirection();
  prd = (.2f + .8f*fabs(dot(rayDir,Ng)))*self.color;
}

OPTIX_MISS_PROGRAM(miss)()
{
  const vec2i pixelID = owl::getLaunchIndex();

  const MissProgData &self = owl::getProgramData<MissProgData>();
  
  vec3f &prd = owl::getPRD<vec3f>();
  int pattern = (pixelID.x / 8) ^ (pixelID.y/8);
  prd = (pattern&1) ? self.color1 : self.color0;
}









// and 'trailer' that defines the types at the end

/* declares the 'TriangleGeomData' type that holds all the variables
   for our triangle mesh */
PYOWL_EXPORT_TYPE(TrianglesGeomData)
PYOWL_EXPORT_VARIABLE(OWL_BUFPTR,TrianglesGeomData,vertex,vertex)
PYOWL_EXPORT_VARIABLE(OWL_BUFPTR,TrianglesGeomData,index,index)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,TrianglesGeomData,color,color)


/* declares the miss prog, and the two variables it uses */
PYOWL_EXPORT_TYPE(MissProgData)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,MissProgData,color0,color0)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,MissProgData,color1,color1)

/* declares the ray-gen program, which in this example includes camera
   and frame buffer */
PYOWL_EXPORT_TYPE(RayGenData)
PYOWL_EXPORT_VARIABLE(OWL_BUFPTR,RayGenData,fbPtr,fbPtr)
PYOWL_EXPORT_VARIABLE(OWL_GROUP, RayGenData,world,world)
PYOWL_EXPORT_VARIABLE(OWL_INT2,  RayGenData,fbSize,fbSize)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,RayGenData,camera_pos,camera_pos)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,RayGenData,camera_dir_00,camera_dir_00)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,RayGenData,camera_dir_du,camera_dir_du)
PYOWL_EXPORT_VARIABLE(OWL_FLOAT3,RayGenData,camera_dir_dv,camera_dir_dv)


