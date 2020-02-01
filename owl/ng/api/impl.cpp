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

#include <owl/owl.h>
#include "APIContext.h"
#include "APIHandle.h"

namespace owl {

#if 1
# define LOG_API_CALL() /* ignore */
#else 
# define LOG_API_CALL() std::cout << "% " << __FUNCTION__ << "(...)" << std::endl;
#endif


#define LOG(message)                            \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl
  
  OWL_API OWLContext owlContextCreate(int32_t *requestedDeviceIDs,
                                      int      numRequestedDevices)
  {
    LOG_API_CALL();
    APIContext::SP context = std::make_shared<APIContext>(requestedDeviceIDs,
                                                          numRequestedDevices);
    LOG("context created...");
    return (OWLContext)context->createHandle(context);
  }

  /*! set number of ray types to be used in this context; this should be
    done before any programs, pipelines, geometries, etc get
    created */
  OWL_API void
  owlContextSetRayTypeCount(OWLContext _context,
                            size_t numRayTypes)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    context->setRayTypeCount(numRayTypes);
  }


  /*! sets maximum instancing depth for the given context:

    '0' means 'no instancing allowed, only bottom-level accels; 
  
    '1' means 'at most one layer of instances' (ie, a two-level scene),
    where the 'root' world rays are traced against can be an instance
    group, but every child in that inscne group is a geometry group.

    'N>1" means "up to N layers of instances are allowed.

    The default instancing depth is 1 (ie, a two-level scene), since
    this allows for most use cases of instancing and is still
    hardware-accelerated. Using a node graph with instancing deeper than
    the configured value will result in wrong results; but be aware that
    using any value > 1 here will come with a cost. It is recommended
    to, if at all possible, leave this value to one and convert the
    input scene to a two-level scene layout (ie, with only one level of
    instances) */
  OWL_API void
  owlSetMaxInstancingDepth(OWLContext _context,
                           int32_t maxInstanceDepth)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    context->setMaxInstancingDepth(maxInstanceDepth);
  }
  
  
  OWL_API void owlBuildSBT(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    context->buildSBT();
  }

  OWL_API void owlBuildPrograms(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    context->buildPrograms();
  }
  
  OWL_API void owlBuildPipeline(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    context->buildPipeline();
  }
  
  OWL_API void owlParamsLaunch2D(OWLRayGen _rayGen,
                                 int dims_x, int dims_y,
                                 OWLLaunchParams _launchParams)
  {
    LOG_API_CALL();

    assert(_rayGen);
    RayGen::SP rayGen
      = ((APIHandle *)_rayGen)->get<RayGen>();
    assert(rayGen);

    assert(_launchParams);
    LaunchParams::SP launchParams
      = ((APIHandle *)_launchParams)->get<LaunchParams>();
    assert(launchParams);

    rayGen->launch(vec2i(dims_x,dims_y),launchParams);
  }

  OWL_API void owlRayGenLaunch2D(OWLRayGen _rayGen,
                                 int dims_x, int dims_y)
  {
    LOG_API_CALL();

    assert(_rayGen);
    RayGen::SP rayGen
      = ((APIHandle *)_rayGen)->get<RayGen>();
    assert(rayGen);

    rayGen->launch(vec2i(dims_x,dims_y));
  }


  OWL_API int32_t owlGetDeviceCount(OWLContext _context)
  {
    LOG_API_CALL();

    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->getContext();
    assert(context);
    return (int32_t)lloGetDeviceCount(context->llo);
  }
    

  
  // ==================================================================
  // <object>::getVariable
  // ==================================================================
  template<typename T>
  OWLVariable
  getVariableHelper(APIHandle *handle,
                    const char *varName)
  {
    assert(varName);
    assert(handle);
    typename T::SP obj = handle->get<T>();
    assert(obj);

    if (!obj->hasVariable(varName))
      throw std::runtime_error("Trying to get reference to variable '"+std::string(varName)+
                               "' on object that does not have such a variable");
    
    Variable::SP var = obj->getVariable(varName);
    assert(var);

    APIContext::SP context = handle->getContext();
    assert(context);

    return(OWLVariable)context->createHandle(var);
  }
  
  
  OWL_API OWLVariable
  owlGeomGetVariable(OWLGeom _geom,
                         const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<Geom>((APIHandle*)_geom,varName);
  }

  OWL_API OWLVariable
  owlRayGenGetVariable(OWLRayGen _prog,
                       const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<RayGen>((APIHandle*)_prog,varName);
  }

  OWL_API OWLVariable
  owlMissProgGetVariable(OWLMissProg _prog,
                       const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<MissProg>((APIHandle*)_prog,varName);
  }

  OWL_API OWLVariable
  owlLaunchParamsGetVariable(OWLLaunchParams _prog,
                       const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<LaunchParams>((APIHandle*)_prog,varName);
  }
  

  std::vector<OWLVarDecl> checkAndPackVariables(const OWLVarDecl *vars,
                                                size_t            numVars)
  {
    if (vars == nullptr && (numVars == 0 || numVars == -1))
      return {};

    // *copy* the vardecls here, so we can catch any potential memory
    // *access errors early

    assert(vars);
    if (numVars == size_t(-1)) {
      // using -1 as count value for a variable list means the list is
      // null-terminated... so just count it
      for (numVars = 0; vars[numVars].name != nullptr; numVars++);
    }
    for (int i=0;i<numVars;i++)
      assert(vars[i].name != nullptr);
    std::vector<OWLVarDecl> varDecls(numVars);
    std::copy(vars,vars+numVars,varDecls.begin());
    return varDecls;
  }

  OWL_API OWLRayGen
  owlRayGenCreate(OWLContext _context,
                  OWLModule  _module,
                  const char *programName,
                  size_t      sizeOfVarStruct,
                  OWLVarDecl *vars,
                  size_t      numVars)
  {
    LOG_API_CALL();

    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    
    assert(_module);
    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);
    
    RayGenType::SP  rayGenType
      = context->createRayGenType(module,programName,
                                  sizeOfVarStruct,
                                  checkAndPackVariables(vars,numVars));
    assert(rayGenType);
    
    RayGen::SP  rayGen
      = context->createRayGen(rayGenType);
    assert(rayGen);
    return (OWLRayGen)context->createHandle(rayGen);
  }


  OWL_API OWLLaunchParams
  owlLaunchParamsCreate(OWLContext _context,
                        size_t      sizeOfVarStruct,
                        OWLVarDecl *vars,
                        size_t      numVars)
  {
    LOG_API_CALL();

    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    
    LaunchParamsType::SP  launchParamsType
      = context->createLaunchParamsType(sizeOfVarStruct,
                                        checkAndPackVariables(vars,numVars));
    assert(launchParamsType);
    
    LaunchParams::SP  launchParams
      = context->createLaunchParams(launchParamsType);
    assert(launchParams);
    return (OWLLaunchParams)context->createHandle(launchParams);
  }



  OWL_API OWLMissProg
  owlMissProgCreate(OWLContext _context,
                    OWLModule  _module,
                    const char *programName,
                    size_t      sizeOfVarStruct,
                    OWLVarDecl *vars,
                    size_t      numVars)
  {
    LOG_API_CALL();

    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    
    assert(_module);
    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);
    
    MissProgType::SP  missProgType
      = context->createMissProgType(module,programName,
                                  sizeOfVarStruct,
                                  checkAndPackVariables(vars,numVars));
    assert(missProgType);
    
    MissProg::SP  missProg
      = context->createMissProg(missProgType);
    assert(missProg);

    return (OWLMissProg)context->createHandle(missProg);
  }

  OWL_API OWLGroup
  owlTrianglesGeomGroupCreate(OWLContext _context,
                              size_t numGeometries,
                              OWLGeom *initValues)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    GeomGroup::SP  group = context->trianglesGeomGroupCreate(numGeometries);
    assert(group);

    OWLGroup _group = (OWLGroup)context->createHandle(group);
    if (initValues) {
      for (int i = 0; i < numGeometries; i++) {
        //owlGeomGroupSetChild(_group, i, initValues[i]);
        Geom::SP child = ((APIHandle *)initValues[i])->get<TrianglesGeom>();
        assert(child);
        group->setChild(i, child);
      }
    }
    assert(_group);
    return _group;
  }

  OWL_API OWLGroup
  owlUserGeomGroupCreate(OWLContext _context,
                         size_t numGeometries,
                         OWLGeom *initValues)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    GeomGroup::SP  group = context->userGeomGroupCreate(numGeometries);
    assert(group);
    
    OWLGroup _group = (OWLGroup)context->createHandle(group);
    if (initValues) {
      for (int i = 0; i < numGeometries; i++) {
        //owlGeomGroupSetChild(_group, i, initValues[i]);
        Geom::SP child = ((APIHandle *)initValues[i])->get<UserGeom>();
        assert(child);
        group->setChild(i, child);
      }
    }
    assert(_group);
    return _group;
  }

  OWL_API OWLGroup
  owlInstanceGroupCreate(OWLContext _context,
                         size_t numInstances)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    InstanceGroup::SP  group = context->createInstanceGroup(numInstances);
    assert(group);

    OWLGroup _group = (OWLGroup)context->createHandle(group);
    assert(_group);
    return _group;
  }




  

  OWL_API void owlContextDestroy(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    
    context->releaseAll();
  }

  OWL_API OWLBuffer
  owlDeviceBufferCreate(OWLContext _context,
                        OWLDataType type,
                        size_t count,
                        const void *init)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    Buffer::SP  buffer  = context->deviceBufferCreate(type,count,init);
    assert(buffer);
    return (OWLBuffer)context->createHandle(buffer);
  }

  OWL_API OWLBuffer
  owlHostPinnedBufferCreate(OWLContext _context,
                            OWLDataType type,
                            size_t      count)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    Buffer::SP  buffer  = context->hostPinnedBufferCreate(type,count);
    assert(buffer);
    return (OWLBuffer)context->createHandle(buffer);
  }

  OWL_API const void *
  owlBufferGetPointer(OWLBuffer _buffer, int deviceID)
  {
    LOG_API_CALL();
    assert(_buffer);
    Buffer::SP buffer = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);
    return buffer->getPointer(deviceID);
  }

  OWL_API CUstream
  owlParamsGetCudaStream(OWLLaunchParams _lp, int deviceID)
  {
    LOG_API_CALL();
    assert(_lp);
    LaunchParams::SP lp = ((APIHandle *)_lp)->get<LaunchParams>();
    assert(lp);
    return lp->getCudaStream(deviceID);
  }

  OWL_API void 
  owlBufferResize(OWLBuffer _buffer, size_t newItemCount)
  {
    LOG_API_CALL();
    assert(_buffer);
    Buffer::SP buffer = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);
    return buffer->resize(newItemCount);
  }

  OWL_API void 
  owlBufferUpload(OWLBuffer _buffer, const void *hostPtr)
  {
    LOG_API_CALL();
    assert(_buffer);
    Buffer::SP buffer = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);
    return buffer->upload(hostPtr);
  }

  /*! destroy the given buffer; this will both release the app's
    refcount on the given buffer handle, *and* the buffer itself; i.e.,
    even if some objects still hold variables that refer to the old
    handle the buffer itself will be freed */
  OWL_API void 
  owlBufferDestroy(OWLBuffer _buffer)
  {
    LOG_API_CALL();
    assert(_buffer);
    APIHandle *handle = (APIHandle *)_buffer;
    assert(handle);
    
    Buffer::SP buffer = handle->get<Buffer>();
    assert(buffer);
    buffer->destroy();

    handle->clear();
  }

  OWL_API OWLGeomType
  owlGeomTypeCreate(OWLContext  _context,
                    OWLGeomKind kind,
                    size_t      varStructSize,
                    OWLVarDecl *vars,
                    size_t      numVars)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    GeomType::SP geometryType
      = context->createGeomType(kind,varStructSize,
                                    checkAndPackVariables(vars,numVars));
    assert(geometryType);
    return (OWLGeomType)context->createHandle(geometryType);
  }
  
  OWL_API OWLGeom
  owlGeomCreate(OWLContext  _context,
                OWLGeomType _geometryType)
  {
    assert(_geometryType);
    assert(_context);

    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);

    GeomType::SP geometryType
      = ((APIHandle *)_geometryType)->get<GeomType>();
    assert(geometryType);

    Geom::SP geometry
      = geometryType->createGeom();
    assert(geometry);

    return (OWLGeom)context->createHandle(geometry);
  }
  
  /*! set the primitive count for the given uesr geometry. this _has_
    to be set before the group(s) that this geom is used in get
    built */
  OWL_API void
  owlGeomSetPrimCount(OWLGeom _geom,
                      size_t  primCount)
  {
    assert(_geom);
    UserGeom::SP geom = ((APIHandle *)_geom)->get<UserGeom>();
    geom->setPrimCount(primCount);
  }

  
  OWL_API OWLModule owlModuleCreate(OWLContext _context,
                                    const char *ptxCode)
  {
    LOG_API_CALL();
    assert(_context);
    assert(ptxCode);
    
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    Module::SP  module  = context->createModule(ptxCode);
    assert(module);
    return (OWLModule)context->createHandle(module);
  }


  // ==================================================================
  // "RELEASE" functions
  // ==================================================================
  template<typename T>
  void releaseObject(APIHandle *handle)
  {
    assert(handle);

    // we don't actually _need_ this object, but let's do this just
    // for sanity's sake
    typename T::SP object = handle->get<T>();
    assert(object);

    delete handle;
  }
  

  OWL_API void owlBufferRelease(OWLBuffer buffer)
  {
    LOG_API_CALL();
    releaseObject<Buffer>((APIHandle*)buffer);
  }
  
  OWL_API void owlModuleRelease(OWLModule module) 
  {
    LOG_API_CALL();
    releaseObject<Module>((APIHandle*)module);
  }
  
  OWL_API void owlGroupRelease(OWLGroup group)
  {
    LOG_API_CALL();
    releaseObject<Group>((APIHandle*)group);
  }
  
  OWL_API void owlRayGenRelease(OWLRayGen handle)
  {
    LOG_API_CALL();
    releaseObject<RayGen>((APIHandle*)handle);
  }
  
  OWL_API void owlVariableRelease(OWLVariable variable)
  {
    LOG_API_CALL();
    releaseObject<Variable>((APIHandle*)variable);
  }
  
  OWL_API void owlGeomRelease(OWLGeom geometry)
  {
    LOG_API_CALL();
    releaseObject<Geom>((APIHandle*)geometry);
  }

  // ==================================================================
  // "Triangles" functions
  // ==================================================================
  OWL_API void
  owlTrianglesSetVertices(OWLGeom   _triangles,
                          OWLBuffer _buffer,
                          size_t count,
                          size_t stride,
                          size_t offset)
  {
    LOG_API_CALL();
    
    assert(_triangles);
    assert(_buffer);

    TrianglesGeom::SP triangles
      = ((APIHandle *)_triangles)->get<TrianglesGeom>();
    assert(triangles);

    Buffer::SP buffer
      = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);

    triangles->setVertices(buffer,count,stride,offset);
  }

  OWL_API void owlGroupBuildAccel(OWLGroup _group)
  {
    LOG_API_CALL();
    
    assert(_group);

    Group::SP group
      = ((APIHandle *)_group)->get<Group>();
    assert(group);
    
    group->buildAccel();
  }  

  OWL_API void
  owlTrianglesSetIndices(OWLGeom   _triangles,
                         OWLBuffer _buffer,
                         size_t count,
                         size_t stride,
                         size_t offset)
  {
    LOG_API_CALL();
    
    assert(_triangles);
    assert(_buffer);

    TrianglesGeom::SP triangles
      = ((APIHandle *)_triangles)->get<TrianglesGeom>();
    assert(triangles);

    Buffer::SP buffer
      = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);

    triangles->setIndices(buffer,count,stride,offset);
  }

  // ==================================================================
  // function pointer setters ....
  // ==================================================================
  OWL_API void
  owlGeomTypeSetClosestHit(OWLGeomType _geometryType,
                           int             rayType,
                           OWLModule       _module,
                           const char     *progName)
  {
    LOG_API_CALL();
    
    assert(_geometryType);
    assert(_module);
    assert(progName);

    GeomType::SP geometryType
      = ((APIHandle *)_geometryType)->get<GeomType>();
    assert(geometryType);

    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);

    geometryType->setClosestHitProgram(rayType,module,progName);
  }

  OWL_API void
  owlGeomTypeSetIntersectProg(OWLGeomType _geometryType,
                           int             rayType,
                           OWLModule       _module,
                           const char     *progName)
  {
    LOG_API_CALL();
    
    assert(_geometryType);
    assert(_module);
    assert(progName);

    UserGeomType::SP geometryType
      = ((APIHandle *)_geometryType)->get<UserGeomType>();
    assert(geometryType);

    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);

    geometryType->setIntersectProg(rayType,module,progName);
  }
  
  OWL_API void
  owlGeomTypeSetBoundsProg(OWLGeomType _geometryType,
                           OWLModule       _module,
                           const char     *progName)
  {
    LOG_API_CALL();
    
    assert(_geometryType);
    assert(_module);
    assert(progName);

    UserGeomType::SP geometryType
      = ((APIHandle *)_geometryType)->get<UserGeomType>();
    assert(geometryType);

    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);

    geometryType->setBoundsProg(module,progName);
  }


  // ==================================================================
  // "VariableSet" functions, for each element type
  // ==================================================================

  template<typename T>
  void setVariable(APIHandle *handle, const T &value)
  {
    assert(handle);

    Variable::SP variable
      = handle->get<Variable>();
    assert(variable);

    variable->set(value);
  }


#if 1

#define _OWL_SET_HELPER(stype,abb)                      \
  OWL_API void owlVariableSet1##abb(OWLVariable var,    \
                                    stype v)            \
  {                                                     \
    LOG_API_CALL();                                     \
    setVariable((APIHandle *)var,v);                    \
  }                                                     \
  OWL_API void owlVariableSet2##abb(OWLVariable var,    \
                                    stype x,            \
                                    stype y)            \
  {                                                     \
    LOG_API_CALL();                                     \
    setVariable((APIHandle *)var,vec2##abb(x,y));       \
  }                                                     \
  OWL_API void owlVariableSet3##abb(OWLVariable var,    \
                                    stype x,            \
                                    stype y,            \
                                    stype z)            \
  {                                                     \
    LOG_API_CALL();                                     \
    setVariable((APIHandle *)var,vec3##abb(x,y,z));     \
  }                                                     \
  OWL_API void owlVariableSet4##abb(OWLVariable var,    \
                                    stype x,            \
                                    stype y,            \
                                    stype z,            \
                                    stype w)            \
  {                                                     \
    LOG_API_CALL();                                     \
    setVariable((APIHandle *)var,vec4##abb(x,y,z,w));   \
  }                                                     \
  /*end of macro */
  _OWL_SET_HELPER(int32_t,i)
  _OWL_SET_HELPER(uint32_t,ui)
  _OWL_SET_HELPER(int64_t,l)
  _OWL_SET_HELPER(uint64_t,ul)
  _OWL_SET_HELPER(float,f)
#undef _OWL_SET_HELPER

#else
  // ----------- set1 -----------
  OWL_API void owlVariableSet1f(OWLVariable _variable, float value)
  {
    LOG_API_CALL();
    setVariable((APIHandle *)_variable,value);
  }

  OWL_API void owlVariableSet1i(OWLVariable _variable, int value)
  {
    LOG_API_CALL();
    setVariable((APIHandle *)_variable,value);
  }
  OWL_API void owlVariableSeti(OWLVariable _variable, int value)
  {
    LOG_API_CALL();
    setVariable((APIHandle *)_variable,value);
  }

  
  // ----------- set2 -----------
  OWL_API void owlVariableSet2i(OWLVariable _variable, int x, int y)
  {
    LOG_API_CALL();
    setVariable((APIHandle *)_variable,vec2i(x,y));
  }

  // ----------- set2v -----------
  OWL_API void owlVariableSet2iv(OWLVariable _variable, const int *value)
  {
    LOG_API_CALL();
    assert(value);
    setVariable((APIHandle *)_variable,*(const vec2i*)value);
  }

  // ----------- set3 -----------
  OWL_API void owlVariableSet3f(OWLVariable _variable,
                                float x, float y, float z)
  {
    LOG_API_CALL();
    setVariable((APIHandle *)_variable,vec3f(x,y,z));
  }
  // ----------- set3v -----------
  OWL_API void owlVariableSet3fv(OWLVariable _variable, const float *value)
  {
    LOG_API_CALL();
    assert(value);
    setVariable((APIHandle *)_variable,*(const vec3f*)value);
  }
#endif
  
  // ----------- set<other> -----------
  OWL_API void owlVariableSetGroup(OWLVariable _variable, OWLGroup _group)
  {
    LOG_API_CALL();

    APIHandle *handle = (APIHandle*)_group;
    Group::SP group
      = handle
      ? handle->get<Group>()
      : Group::SP();
    
    assert(group);

    setVariable((APIHandle *)_variable,group);
  }

  OWL_API void owlVariableSetBuffer(OWLVariable _variable, OWLBuffer _buffer)
  {
    LOG_API_CALL();

    APIHandle *handle = (APIHandle*)_buffer;
    Buffer::SP buffer
      = handle
      ? handle->get<Buffer>()
      : Buffer::SP();

    setVariable((APIHandle *)_variable,buffer);
  }

  OWL_API void owlVariableSetRaw(OWLVariable _variable, const void *valuePtr)
  {
    LOG_API_CALL();

    APIHandle *handle = (APIHandle*)_variable;
    assert(handle);

    Variable::SP variable
      = handle->get<Variable>();
    assert(variable);

    variable->setRaw(valuePtr);
  }

  // -------------------------------------------------------
  // group/hierarchy creation and setting
  // -------------------------------------------------------
  OWL_API void
  owlInstanceGroupSetChild(OWLGroup _group,
                           int whichChild,
                           OWLGroup _child)
  {
    LOG_API_CALL();

    assert(_group);
    InstanceGroup::SP group = ((APIHandle*)_group)->get<InstanceGroup>();
    assert(group);

    assert(_child);
    Group::SP child = ((APIHandle *)_child)->get<Group>();
    assert(child);

    group->setChild(whichChild, child);
  }

  OWL_API void
  owlInstanceGroupSetTransform(OWLGroup _group,
                               int whichChild,
                               const float *floats,
                               OWLMatrixFormat matrixFormat)
  {
    LOG_API_CALL();

    assert("check for valid transform" && floats != nullptr);
    affine3f xfm;
    switch(matrixFormat) {
    case OWL_MATRIX_FORMAT_OWL:
      xfm = *(const affine3f*)floats;
      break;
    case OWL_MATRIX_FORMAT_ROW_MAJOR:
      xfm.l.vx = vec3f(floats[0+0],floats[4+0],floats[8+0]);
      xfm.l.vy = vec3f(floats[0+1],floats[4+1],floats[8+1]);
      xfm.l.vz = vec3f(floats[0+2],floats[4+2],floats[8+2]);
      xfm.p    = vec3f(floats[0+3],floats[4+3],floats[8+3]);
      break;
    default:
      throw std::runtime_error("un-recognized matrix format");
    }
    
    assert(_group);
    InstanceGroup::SP group = ((APIHandle*)_group)->get<InstanceGroup>();
    assert(group);

    group->setTransform(whichChild, xfm);
  }


} // ::owl
