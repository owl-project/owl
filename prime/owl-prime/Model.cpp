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

#include "owl-prime/Model.h"
#include "owl-prime/Context.h"
#include "owl-prime/Group.h"
#include "primer-common/StridedArray.h"

namespace op {
  using primer::StridedArray;
  
  Model::Model(Context *context,
               const std::vector<OPGroup>  &groups,
               const std::vector<affine3f> &xfms)
    : // primer::Model(context),
      context(context)
  {
    // std::vector<affine3f> owlXfms;
    std::vector<OWLGroup> owlGroups;
    for (auto &g : groups) {
      op::Group *group = (op::Group *)g;
      if (group->trianglesGroup) {
        owlGroups.push_back(group->trianglesGroup);
      }
      if (group->userGroup) {
        owlGroups.push_back(group->userGroup);
      }
    }
    handle
      = owlInstanceGroupCreate(context->owl,
                               owlGroups.size(),
                               owlGroups.data(),
                               nullptr,
                               (const float *)xfms.data());
    owlGroupBuildAccel(handle);
  }

  void Model::trace(Ray *rays,
                    Hit *hits,
                    int  numRays,
                    int *activeIDs,
                    int  numActive,
                    OPTraceFlags flags)
  {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    context->checkSBT();
    
    auto &lp = context->launchParams;
    const int numHits = numRays;
    
    Ray *d_rays = 0;
    Hit *d_hits = 0;
    int *d_activeIDs = 0;
    
    if (isDeviceAccessible(rays)) {
      d_rays = rays;
    } else {
      CUDA_CALL(Malloc(&d_rays,numRays*sizeof(Ray)));
      CUDA_CALL(Memcpy(d_rays,rays,numRays*sizeof(Ray),cudaMemcpyDefault));
    }
    if (isDeviceAccessible(hits)) {
      d_hits = hits;
    } else {
      CUDA_CALL(Malloc(&d_hits,numRays*sizeof(Hit)));
      CUDA_CALL(Memcpy(d_hits,hits,numRays*sizeof(Hit),cudaMemcpyDefault));
    }
    if (isDeviceAccessible(activeIDs)) {
      d_activeIDs = activeIDs;
    } else {
      CUDA_CALL(Malloc(&d_activeIDs,numActive*sizeof(int)));
      CUDA_CALL(Memcpy(d_activeIDs,activeIDs,numActive*sizeof(int),cudaMemcpyDefault));
    }
    owlParamsSetPointer(lp,"rays",d_rays);
    owlParamsSetPointer(lp,"hits",d_hits);
    owlParamsSetPointer(lp,"activeIDs",d_activeIDs);
    owlParamsSet1i(lp,"numRays",numActive?numActive:numRays);
    owlParamsSet1ul(lp,"flags",flags);
    owlParamsSetGroup(lp,"model",handle);

    owlLaunch2D(context->rayGen,numRays,1,lp);
    
    if (d_hits != hits) {
      CUDA_CALL(Memcpy(hits,d_hits,numRays*sizeof(Hit),cudaMemcpyDefault));
    }
    
    if (d_rays != rays)
      CUDA_CALL(Free(d_rays));
    if (d_hits != hits)
      CUDA_CALL(Free(d_hits));
    if (d_activeIDs != activeIDs)
      CUDA_CALL(Free(d_activeIDs));
    
    CUDA_SYNC_CHECK();
  }

  void Model::build() 
  { /* nothing - build happened upon creation */ }

} // ::op
