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

#include "primer-common/Context.h"
#include "owl-prime/Triangles.h"
#include "owl-prime/Model.h"
#include "owl-prime/Group.h"

#define CUDA_CALL(a) OWL_CUDA_CALL(a)
#define CUDA_SYNC_CHECK OWL_CUDA_SYNC_CHECK

namespace op {
  using namespace owl::common;
  
  inline bool isDeviceAccessible(const void *ptr)
  {
    cudaPointerAttributes attributes = {};
    // do NOT check for error: in CUDA<10, passing a host pointer will
    // actually create a cuda error, so let's just call and then
    // ignore/clear the error
    cudaPointerGetAttributes(&attributes,ptr);
    cudaGetLastError();

    return attributes.devicePointer != 0;
  }
  
  struct Context : public primer::Context {
    /*! launch parameters */
    struct LPData
    {
      OPTraceFlags           flags;
      OptixTraversableHandle model;
      int                    numRays;

      // ------------------------------------------------------------------
      // array-of-structs variants; when tracing soa rays there may be
      // null or invalid (do NOT put into a union - unions aren't
      // supported by owl variabels)
      // ------------------------------------------------------------------
      primer::Ray *rays;
      primer::Hit *hits;
      int         *activeIDs;
    };
    
    Context(int gpuID);

    static OWLVarDecl lpVariables[];

    /*! create a mesh from vertex array and index array */
    primer::Geom *createTriangles(uint64_t userID,
                                  /* vertex array */
                                  const vec3f *vertices,
                                  size_t numVertices,
                                  size_t sizeOfVertexInBytes,
                                  /* index array */
                                  const vec3i *indices,
                                  size_t numIndices,
                                  size_t sizeOfIndesStructInBytes) override;
    
    primer::Model *createModel(const std::vector<OPGroup>  &groups,
                               const std::vector<affine3f> &xfms) override;
    
    primer::Group *createGroup(std::vector<OPGeom> &geoms) override;
    
    void checkSBT();
    
    bool        sbtDirty        = true;
    OWLContext  owl             = 0;
    OWLGeomType meshGeomType    = 0;
    OWLGeomType spheresGeomType = 0;
    OWLParams   launchParams    = 0;
    OWLRayGen   rayGen          = 0;
    OWLModule   module          = 0;
  };
  
}
