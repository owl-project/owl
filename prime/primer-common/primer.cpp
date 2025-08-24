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

/*! \file primer/common/primer.cpp Implements the 'primer' API */

// the public API that we're implementing in this file ...
#include <owl/owl_prime.h>
// for our implementation
#include "primer-common/Context.h"
#include "primer-common/Model.h"

#define WARN(a) std::cout << OWL_TERMINAL_RED << a << OWL_TERMINAL_DEFAULT << std::endl;

namespace primer {

  /*! enables logging/debugging messages if true, suppresses them if
    false */
#ifdef NDEBUG
  bool Context::logging = false;
#else
  bool Context::logging = true;
#endif
  
#if PRIMER_HAVE_EMBREE
  /* the "real" Context *Context::createEmbreeContext() will be
     defined in embree/Context.cpp */
#else
  Context *Context::createEmbreeContext()
  { 
    throw std::runtime_error("user explicitly requested CPU context, but this library was compiled without embree support");
  }
#endif

  Geom *castCheck(OPGeom _geom)
  {
    assert(_geom);
    return (Geom *)_geom;
  }

  Group *castCheck(OPGroup _group)
  {
    assert(_group);
    return (Group *)_group;
  }

  Model *castCheck(OPModel _model)
  {
    assert(_model);
    return (Model *)_model;
  }

  Context *castCheck(OPContext _context)
  {
    assert(_context);
    return (Context *)_context;
  }
  
}

using namespace primer;

/*! trace all the rays in the given array of input rays, and write
 *  results into the given array of output rays. Unlike
 *  opTraceAsync, this call is synchronous, so will block the
 *  calling thread until all rays are traced and all hits have been
 *  written. Input and output arrays have to remain valid and
 *  unmodified while teh call is active, but can be modified or
 *  released the moment this call returns */
OP_API void opTrace(/*! the model to trace into */
                    OPModel model,
                    /*! number of rays to trace - both arrays must have as
                     *  many entries */
                    size_t numRays,
                    /*! array of rays; must be (at least) as many as
                     *  'numRays' */
                    OPRay *arrayOfRays,
                    /*! array of where to write the results; as many as
                     *  'numRays' */
                    OPHit *arrayOfHits,
                    /*! trace flags that can fine-tune how the trace
                     *  executes. */
                    OPTraceFlags flags)
{
  assert(model);
  if (numRays == 0) return;
  castCheck(model)->trace((primer::Ray *)arrayOfRays,
                          (primer::Hit *)arrayOfHits,
                          (int)numRays,
                          0,0,
                          flags);
}

OP_API void opTraceIndexed(/*! the model to trace into */
                           OPModel model,
                           /*! number of ray indices - this is the
                             number of rays to be traced. */
                           size_t   numActiveRayIndices,
                           /*! optionally, a array of ray IDs that specify which
                            *  of the rays in 'arrayOfRays' to trace. if null, we
                            *  impliictly assume the list of active rays/hits is
                            *  [0,1,2,...numRays) */
                           int32_t *arrayOfActiveRayIndices,

                           /*! size of ray and hit arrays, required
                             for being able to offload those
                             arrays if required */
                           size_t sizeOfRayAndHitArray,

                           /*! array of rays; must be (at least) as many as
                            *  'numRays' */
                           OPRay *arrayOfRays,
                           /*! array of where to write the results; as many as
                            *  'numRays' */
                           OPHit *arrayOfHits,
                           /*! trace flags that can fine-tune how the trace
                            *  executes. */
                           OPTraceFlags flags) 
{
  assert(model);
  castCheck(model)->trace((primer::Ray *)arrayOfRays,
                          (primer::Hit *)arrayOfHits,
                          (int)sizeOfRayAndHitArray,
                          arrayOfActiveRayIndices,
                          (int)numActiveRayIndices,
                          flags);
}

OP_API OPContext opContextCreate(OPContextType contextType,
                                 /*! which GPU to use, '0' being the first
                                  *  GPU, '1' the second, etc. '-1' means
                                  *  'use host CPU only' */
                                 int32_t gpuToUse)
{
  if (contextType == OP_CONTEXT_HOST_FORCE_CPU) {
    return (OPContext)primer::Context::createEmbreeContext();
  }

  try {
    return (OPContext)primer::Context::createOffloadContext(gpuToUse);
  } catch (...) {
    if (contextType == OP_CONTEXT_REQUIRE_GPU) {
      WARN("could not create required GPU context");
      //throw std::runtime_error("could not create required GPU context");
      return (OPContext)0;
    }
    WARN("could not create a GPU context; falling back to CPU context");
    return (OPContext)primer::Context::createEmbreeContext();
  }
}

OP_API
OPGeom opMeshCreate(OPContext _context,
                    uint64_t userGeomIDtoUseInHits,
                    /* vertex array */
                    const float *vertices,
                    size_t numVertices,
                    size_t sizeOfVertexInBytes,
                    /* index array */
                    const int   *indices,
                    size_t numIndices,
                    size_t sizeOfIndexStructInBytes)
{
  Context *context = castCheck(_context);
  Geom *mesh = context->createTriangles
    (/* ID */userGeomIDtoUseInHits,
     /* vertex array */
     (const vec3f*)vertices,numVertices,sizeOfVertexInBytes,
     /* index array */
     (const vec3i*)indices,numIndices,sizeOfIndexStructInBytes);
  assert(mesh);
  return (OPGeom)mesh;
}
  
OP_API OPModel opModelFromGeoms(OPContext _context,
                                OPGeom *geoms,
                                size_t numGeoms)
{
  affine3f xfm;
  OPGroup group = opGroupCreate(_context,geoms,1);
  return opModelCreate(_context,&group,(OPTransform*)&xfm,1);
}

OP_API OPModel opModelFromGeom(OPContext context,
                               OPGeom mesh)
{
  return opModelFromGeoms(context,&mesh,1);
}

OP_API
OPGroup opGroupCreate(OPContext _context,
                      OPGeom *_geoms,
                      int numGeoms)
{
  Context *context = castCheck(_context);
  std::vector<OPGeom> geoms;
  for (int i=0;i<numGeoms;i++) {
    assert(_geoms[i]);
    geoms.push_back(_geoms[i]);
  }
  Group *group = context->createGroup(geoms);
  assert(group);
  return (OPGroup)group;
}

OP_API
OPModel opModelCreate(OPContext    _context,
                      OPGroup     *groups,
                      OPTransform *xfms,
                      int          numInstances)
{
  Context *context = castCheck(_context);
  
  std::vector<affine3f> _xfms(numInstances);
  if (xfms) 
    std::copy(xfms,xfms+numInstances,(OPTransform*)_xfms.data());
  else {
    for (int i=0;i<numInstances;i++)
      _xfms[i] = owl::common::affine3f();
  }

  std::vector<OPGroup> _groups(numInstances);
  std::copy(groups,groups+numInstances,_groups.data());

  Model *model = context->createModel(_groups,_xfms);
  assert(model);
  return (OPModel)model;
}


