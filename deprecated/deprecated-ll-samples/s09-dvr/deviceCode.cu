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

// Ray gen shader for ll00-rayGenOnly. No actual rays are harmed in the making of
// this shader. The pixel location is simply translated into a checkerboard pattern.

#include "deviceCode.h"
#include <optix_device.h>


inline __device__ float transferFunction(float value)
{
  return powf(fabsf(value-.01f),3.f);
}

inline __device__ vec3f traceRay(const owl::Ray &ray,
                                 const vec3i &dims,
                                 const float *voxels)
{
  vec3f color = 0.f;
  float opacity = 1.f;

  float t0 = ray.tmin, t1 = ray.tmax;
  const vec3f bounds_lo(0.f);
  const vec3f bounds_hi(1.f);
  const vec3f t_lo = (bounds_lo - ray.origin) / ray.direction;
  const vec3f t_hi = (bounds_hi - ray.origin) / ray.direction;
  t0 = max(t0,reduce_max(min(t_lo,t_hi)));
  t1 = min(t1,reduce_min(max(t_lo,t_hi)));

  const float dt = .001f;
  
  for (float t = t0+.5f*dt; t < t1; t += dt) {
    vec3i coord = vec3i((ray.origin + t * ray.direction) * vec3f(dims));
    coord = min(coord,dims-vec3i(1));
    coord = max(coord,vec3i(0));
    size_t idx = coord.x + size_t(dims.x)*(coord.y+size_t(dims.y)*(coord.z));
    float  val = voxels[idx];
    float alpha = transferFunction(val);
    color += opacity * alpha * dt * vec3f(1.f);
    opacity = opacity * (1.f - alpha * dt);
  }
  
  return color;
}

inline __device__ float fixZeroes(float v)
{
  return fabsf(v) < 1e-6f ? 1e-6f : v;
}
  
inline __device__ vec3f fixZeroes(vec3f v)
{
  return vec3f(fixZeroes(v.x),
               fixZeroes(v.y),
               fixZeroes(v.z));
}
// OPTIX_RAYGEN_PROGRAM() is a simple macro defined in deviceAPI.h to add standard
// code for defining a shader method.
// It puts:
//   extern "C" __global__ void __raygen__##programName
// in front of the program name given
OPTIX_RAYGEN_PROGRAM(renderDVR)()
{
  const RayGenData &self = owl::getProgramData<RayGenData>();
  const vec2i pixelID = owl::getLaunchIndex();
  if (pixelID == owl::vec2i(0)) {
    printf("%sHello OptiX From your First RayGen Program%s\n",
           OWL_TERMINAL_CYAN,
           OWL_TERMINAL_DEFAULT);
  }

  const vec2f screen = (vec2f(pixelID)+vec2f(.5f)) / vec2f(self.fbSize);
  owl::Ray ray;
  ray.origin    
    = self.camera.pos;
  ray.direction 
    = normalize(fixZeroes(self.camera.dir_00
                          + screen.u * self.camera.dir_du
                          + screen.v * self.camera.dir_dv));

  vec3f color = traceRay(ray,self.dims,self.voxels);

  const int fbOfs = pixelID.x+self.fbSize.x*pixelID.y;
  self.fbPtr[fbOfs]
    = owl::make_rgba(color);
}

