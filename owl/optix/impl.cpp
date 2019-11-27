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

#include "Context.h"

namespace owl {
  
#define LOG_API_CALL() std::cout << "% " << __FUNCTION__ << "(...)" << std::endl;

  struct APIContext;
  
  struct APIHandle {
    APIHandle(Object::SP object, APIContext *context);
    virtual ~APIHandle();
    template<typename T> inline std::shared_ptr<T> get();
    inline std::shared_ptr<APIContext> getContext() const { return context; }
    inline bool isContext() const
    {
      return ((void*)object.get() == (void*)context.get());
    }
    std::string toString() const
    {
      assert(object);
      return object->toString();
    }
    std::shared_ptr<Object>     object;
    std::shared_ptr<APIContext> context;
  };
  

  
  
  struct APIContext : public Context {
    typedef std::shared_ptr<APIContext> SP;
    
    APIHandle *createHandle(Object::SP object);

    void track(APIHandle *object);
    
    void forget(APIHandle *object);

    /*! delete - and thereby, release - all handles that we still
      own. */
    void releaseAll();
    std::set<APIHandle *> activeHandles;
  };
  
  
  void APIContext::forget(APIHandle *object)
  {
    assert(object);
    auto it = activeHandles.find(object);
    assert(it != activeHandles.end());
    activeHandles.erase(it);
  }

  void APIContext::releaseAll()
  {
    std::cout << "#owl: context is dying, num api handles (other than context itself) "
              << "that have not yet been released: "
              << (activeHandles.size()-1)
              << std::endl;
    for (auto handle : activeHandles)
      std::cout << " - " << handle->toString() << std::endl;

    // create a COPY of the handles we need to destroy, else
    // destroying the handles modifies the std::set while we're
    // iterating through it!
    std::set<APIHandle *> stillActiveHandles = activeHandles;
    for (auto handle : stillActiveHandles)  {
      assert(handle);
      delete handle;
    }

    assert(activeHandles.empty());
  }
  
  void APIContext::track(APIHandle *object)
  {
    assert(object);
      
    auto it = activeHandles.find(object);
    assert(it == activeHandles.end());
    activeHandles.insert(object);
  }

  APIHandle *APIContext::createHandle(Object::SP object)
  {
    assert(object);
    APIHandle *handle = new APIHandle(object,this);
    track(handle);
    return handle;
  }

  
  template<typename T> inline std::shared_ptr<T> APIHandle::get()
  {
    assert(object);
    std::shared_ptr<T> asT = std::dynamic_pointer_cast<T>(object);
    if (object && !asT) {
      const std::string objectTypeID = typeid(*object.get()).name();
	
      const std::string tTypeID = typeid(T).name();
      throw std::runtime_error("could not convert APIHandle of type "
                               + objectTypeID
                               + " to object of type "
                               + tTypeID);
    }
    assert(asT);
    return asT;
  }
    

  
  APIHandle::APIHandle(Object::SP object, APIContext *context)
  {
    assert(object);
    assert(context);
    this->object  = object;
    this->context = std::dynamic_pointer_cast<APIContext>
      (context->shared_from_this());
    assert(this->object);
    assert(this->context);
  }

  APIHandle::~APIHandle()
  {
    context->forget(this);
    object  = nullptr;
    context = nullptr;
  }
  
  OWL_API OWLContext owlContextCreate()
  {
    LOG_API_CALL();
    APIContext::SP context = std::make_shared<APIContext>();
    std::cout << "#owl.api: context created..." << std::endl;
    return (OWLContext)context->createHandle(context);
  }

  OWL_API void owlContextLaunch2D(OWLContext _context,
                                  OWLRayGen _rayGen,
                                  int dims_x, int dims_y)
  {
    LOG_API_CALL();

    assert(_context);
    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);

    std::cout << "SHOULD BUILD SBT HERE!!!!" << std::endl;
    context->expBuildSBT();
    
    std::cout << "actual launch not yet implemented - ignoring ...." << std::endl;
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
  owlGeometryGetVariable(OWLGeometry _geom,
                         const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<Geometry>((APIHandle*)_geom,varName);
  }

  OWL_API OWLVariable
  owlRayGenGetVariable(OWLRayGen _prog,
                       const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<RayGen>((APIHandle*)_prog,varName);
  }
  

  std::vector<OWLVarDecl> checkAndPackVariables(const OWLVarDecl *vars,
                                                size_t      numVars)
  {
    // *copy* the vardecls here, so we can catch any potential memory
    // *access errors early
    assert(vars);
    for (int i=0;i<numVars;i++)
      assert(vars[i].name != nullptr);
    std::vector<OWLVarDecl> varDecls(numVars);
    std::copy(vars,vars+numVars,varDecls.begin());
    return varDecls;
  }

  OWL_API OWLRayGen
  owlContextCreateRayGen(OWLContext _context,
                         OWLModule _module,
                         const char *programName,
                         size_t sizeOfVarStruct,
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

  OWL_API OWLGeometryGroup
  owlContextCreateGeometryGroup(OWLContext _context,
                                size_t numGeometries,
                                OWLGeometry *initValues)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    GeometryGroup::SP  group = context->createGeometryGroup(numGeometries);
    assert(group);

    OWLGeometryGroup _group = (OWLGeometryGroup)context->createHandle(group);
    if (initValues) {
      for (int i = 0; i < numGeometries; i++)
        //owlGeometryGroupSetChild(_group, i, initValues[i]);
        group->setChild(i, ((APIHandle *)initValues[i])->get<Geometry>());
    }
    return _group;
  }

  OWL_API OWLInstanceGroup
  owlContextCreateInstanceGroup(OWLContext _context,
                                size_t numInstances)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    InstanceGroup::SP  group = context->createInstanceGroup(numInstances);
    assert(group);

    OWLInstanceGroup _group = (OWLInstanceGroup)context->createHandle(group);
    return _group;
  }




  

  OWL_API void owlContextDestroy(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    
    // Context *context = (Context *)_context;
    context->releaseAll();
    // delete _context;
  }

  OWL_API OWLBuffer
  owlContextCreateBuffer(OWLContext _context,
                         OWLDataType type,
                         int num,
                         const void *init)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    Buffer::SP  buffer  = context->createBuffer();
    assert(buffer);
    return (OWLBuffer)context->createHandle(buffer);
  }

  OWL_API OWLGeometryType
  owlContextCreateGeometryType(OWLContext _context,
                               OWLGeometryKind kind,
                               size_t varStructSize,
                               OWLVarDecl *vars,
                               size_t      numVars)
  {
    LOG_API_CALL();
    assert(_context);
    APIContext::SP context = ((APIHandle *)_context)->get<APIContext>();
    assert(context);
    GeometryType::SP geometryType
      = context->createGeometryType(kind,varStructSize,
                                    checkAndPackVariables(vars,numVars));
    assert(geometryType);
    return (OWLGeometryType)context->createHandle(geometryType);
  }

  OWL_API OWLGeometry
  owlContextCreateGeometry(OWLContext      _context,
                           OWLGeometryType _geometryType)
  {
    assert(_geometryType);
    assert(_context);

    APIContext::SP context
      = ((APIHandle *)_context)->get<APIContext>();
    assert(context);

    GeometryType::SP geometryType
      = ((APIHandle *)_geometryType)->get<GeometryType>();
    assert(geometryType);

    Geometry::SP geometry
      = geometryType->createGeometry();
    assert(geometry);

    return (OWLGeometry)context->createHandle(geometry);
  }

  
  OWL_API OWLModule owlContextCreateModule(OWLContext _context,
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
  
  OWL_API void owlGeometryRelease(OWLGeometry geometry)
  {
    LOG_API_CALL();
    releaseObject<Geometry>((APIHandle*)geometry);
  }

  // ==================================================================
  // "Triangles" functions
  // ==================================================================
  OWL_API void
  owlTrianglesSetVertices(OWLGeometry  _triangles,
                          OWLBuffer    _buffer)
  {
    LOG_API_CALL();
    
    assert(_triangles);
    assert(_buffer);

    TrianglesGeometry::SP triangles
      = ((APIHandle *)_triangles)->get<TrianglesGeometry>();
    assert(triangles);

    Buffer::SP buffer
      = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);

    triangles->setVertices(buffer);
  }

  OWL_API void
  owlTrianglesSetIndices(OWLGeometry  _triangles,
                         OWLBuffer    _buffer)
  {
    LOG_API_CALL();
    
    assert(_triangles);
    assert(_buffer);

    TrianglesGeometry::SP triangles
      = ((APIHandle *)_triangles)->get<TrianglesGeometry>();
    assert(triangles);

    Buffer::SP buffer
      = ((APIHandle *)_buffer)->get<Buffer>();
    assert(buffer);

    triangles->setIndices(buffer);
  }

  // ==================================================================
  // function pointer setters ....
  // ==================================================================
  OWL_API void
  owlGeometryTypeSetClosestHitProgram(OWLGeometryType _geometryType,
                                      int             rayType,
                                      OWLModule       _module,
                                      const char     *progName)
  {
    LOG_API_CALL();
    
    assert(_geometryType);
    assert(_module);
    assert(progName);

    GeometryType::SP geometryType
      = ((APIHandle *)_geometryType)->get<GeometryType>();
    assert(geometryType);

    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);

    geometryType->setClosestHitProgram(rayType,module,progName);
  }


  // ==================================================================
  // "VariableSet" functions, for each element type
  // ==================================================================

  template<typename T>
  void setBasicTypeVariable(APIHandle *handle, const T &value)
  {
    assert(handle);

    Variable::SP variable
      = handle->get<Variable>();
    assert(variable);

    variable->set(value);
  }

  
  OWL_API void owlVariableSet1f(OWLVariable _variable, const float value)
  {
    LOG_API_CALL();
    setBasicTypeVariable((APIHandle *)_variable,(float)value);
  }

  OWL_API void owlVariableSet3fv(OWLVariable _variable, const float *value)
  {
    LOG_API_CALL();
    assert(value);
    setBasicTypeVariable((APIHandle *)_variable,*(const vec3f*)value);
  }
  
  // -------------------------------------------------------
  // group/hierarchy creation and setting
  // -------------------------------------------------------
  OWL_API void
  owlInstanceGroupSetChild(OWLInstanceGroup _group,
                           int whichChild,
                           OWLGeometryGroup _child)
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



  template<typename T>
  struct VariableT : public Variable {
    typedef std::shared_ptr<VariableT<T>> SP;

    VariableT(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    
    void set(const T &value) override { PING; }
  };

  struct BufferPointerVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    BufferPointerVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    Buffer::SP buffer;
  };
  
  Variable::SP Variable::createInstanceOf(const OWLVarDecl *decl)
  {
    assert(decl);
    assert(decl->name);
    switch(decl->type) {
    case OWL_FLOAT:
      return std::make_shared<VariableT<float>>(decl);
    case OWL_FLOAT3:
      return std::make_shared<VariableT<vec3f>>(decl);
    case OWL_BUFFER_POINTER:
      return std::make_shared<BufferPointerVariable>(decl);
    }
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                             +": not yet implemented for type "
                             +typeToString(decl->type));
  }
    

  /*! create one instance each of a given type's variables */
  std::vector<Variable::SP> SBTObjectType::instantiateVariables()
  {
    std::vector<Variable::SP> variables(varDecls.size());
    for (size_t i=0;i<varDecls.size();i++) {
      variables[i] = Variable::createInstanceOf(&varDecls[i]);
      assert(variables[i]);
    }
    return variables;
  }
  

} // ::owl
