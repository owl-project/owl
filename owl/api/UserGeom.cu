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

#include "api/UserGeom.h"
#include "api/Context.h"
#include "ll/DeviceMemory.h"
#include "ll/Device.h"
#include "ll/DeviceGroup.h"
#include "Context.h"

namespace owl {

  __device__ static float atomicMax(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __device__ static float atomicMin(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __global__ void computeBoundsOfPrimBounds(box3f *d_bounds,
                                            const box3f *d_primBounds,
                                            size_t count)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= count) return;

    box3f box = d_primBounds[tid];
    if (!box.empty()) {
      atomicMin(&d_bounds->lower.x,box.lower.x);
      atomicMin(&d_bounds->lower.y,box.lower.y);
      atomicMin(&d_bounds->lower.z,box.lower.z);
      atomicMax(&d_bounds->upper.x,box.upper.x);
      atomicMax(&d_bounds->upper.y,box.upper.y);
      atomicMax(&d_bounds->upper.z,box.upper.z);
    }
  }
                                          
  /*! call a cuda kernel that computes the bounds of the vertex buffers */
  void UserGeom::computeBounds(box3f bounds[2])
  {
    int numThreads = 1024;
    int numBlocks = (primCount + numThreads - 1) / numThreads;

    int oldActive = context->llo->devices[0]->pushActive();
      
    ll::DeviceMemory d_bounds;
    d_bounds.alloc(sizeof(box3f));
    bounds[0] = bounds[1] = box3f();
    d_bounds.upload(bounds);

    ll::UserGeom *ug =
      (ll::UserGeom *)context->llo->devices[0]->checkGetGeom(this->ID);
    
    computeBoundsOfPrimBounds<<<numBlocks,numThreads>>>
      (((box3f*)d_bounds.get())+0,
       (box3f *)ug->internalBufferForBoundsProgram.get(),
       primCount);
    CUDA_SYNC_CHECK();
    d_bounds.download(&bounds[0]);
    d_bounds.free();
    CUDA_SYNC_CHECK();
    bounds[1] = bounds[0];
  }

  UserGeomType::UserGeomType(Context *const context,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls),
      intersectProg(context->numRayTypes)
  {
    /*! nothing special - all inherited */
  }

  UserGeom::UserGeom(Context *const context,
                     GeomType::SP geometryType)
    : Geom(context,geometryType)
  {
    int numPrims = 0;
    context->llo->userGeomCreate(this->ID,geometryType->ID,numPrims);
  }

  void UserGeom::setPrimCount(size_t count)
  {
    primCount = count;
  }

  void UserGeomType::setIntersectProg(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < intersectProg.size());
    intersectProg[rayType].progName = progName;
    intersectProg[rayType].module   = module;
    context->llo->setGeomTypeIntersect(this->ID,
                         rayType,module->ID,
                         // warning: this 'this' here is importat, since
                         // *we* manage the lifetime of this string, and
                         // the one on the constructor list will go out of
                          // scope after this function
                         intersectProg[rayType].progName.c_str());
  }

  void UserGeomType::setBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->boundsProg.progName = progName;
    this->boundsProg.module   = module;
    context->llo->setGeomTypeBoundsProgDevice(this->ID,
                                module->ID,
                                // warning: this 'this' here is importat, since
                                // *we* manage the lifetime of this string, and
                                // the one on the constructor list will go out of
                                // scope after this function
                                this->boundsProg.progName.c_str(),
                                varStructSize
                                );
  }

} // ::owl
