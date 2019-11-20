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

#include "owl/owl.h"
#include "optix/common.h"
#include <set>
#include <map>
#include <typeinfo>

namespace owl {
  using gdt::vec3f;

  struct Context;
  struct Buffer;
  
#define LOG_API_CALL() std::cout << "% " << __FUNCTION__ << "(...)" << std::endl;

#define IGNORING_THIS() std::cout << "## ignoring " << __PRETTY_FUNCTION__ << std::endl;
  
  
//#define OWL_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" : not implemented")
#define OWL_NOTIMPLEMENTED std::cerr << (std::string(__PRETTY_FUNCTION__)+" : not implemented") << std::endl; exit(1);

  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;
    virtual ~Object() {}
  };

  struct BoundVariable;
  
  struct AbstractVariable : public Object {

    typedef std::shared_ptr<AbstractVariable> SP;

    virtual std::shared_ptr<BoundVariable>
    createSetter(Object::SP objectToSetIn) = 0;
  };

  
  template<typename T>
  struct AbstractVariableT : public AbstractVariable {
    typedef std::shared_ptr<AbstractVariableT<T>> SP;
    
    virtual std::shared_ptr<BoundVariable>
    createSetter(Object::SP objectToSetIn) override;
  };

  struct BufferVariable : public AbstractVariable {
    typedef std::shared_ptr<BufferVariable> SP;
    
    virtual std::shared_ptr<BoundVariable>
    createSetter(Object::SP objectToSetIn) override;
  };
  
  struct BoundVariable : public Object {
    typedef std::shared_ptr<BoundVariable> SP;

    virtual void set(const std::shared_ptr<Buffer> &value) { wrongType(); }
    virtual void set(const float &value) { wrongType(); }
    virtual void set(const vec3f &value) { wrongType(); }

    void wrongType() { PING; }
    
    //! the object we're modifying ...
    Object::SP             object;
    //! the variable we're setting in the given object
    AbstractVariable::SP variable;
  };
  
  template<typename T>
  struct BoundVariableT : public BoundVariable {
    typedef std::shared_ptr<BoundVariableT<T>> SP;

    BoundVariableT(AbstractVariable::SP variable,
                   Object::SP object)
    {}
    
    void set(const T &value) override { PING; }
    // T value;
  };

  template<typename T>
  std::shared_ptr<BoundVariable>
  AbstractVariableT<T>::createSetter(Object::SP objectToSet)
  {
    AbstractVariable::SP self
      = std::dynamic_pointer_cast<AbstractVariable>(shared_from_this());
    return std::make_shared<BoundVariableT<T>>(self,objectToSet);
  }

  std::shared_ptr<BoundVariable>
  BufferVariable::createSetter(Object::SP objectToSet)
  {
    AbstractVariable::SP self
      = std::dynamic_pointer_cast<AbstractVariable>(shared_from_this());
    return std::make_shared<BoundVariableT<std::shared_ptr<Buffer>>>(self,objectToSet);
  }
  
  /*! captures the concept of a module that contains one or more
    programs. */
  struct Module : public Object {
    typedef std::shared_ptr<Module> SP;

    Module(const std::string &ptxCode)
      : ptxCode(ptxCode)
    {
      std::cout << "#owl: created module ..." << std::endl;
    }
    
    const std::string ptxCode;
  };
  
  struct SBTObjectType : public Object
  {
    typedef std::shared_ptr<SBTObjectType> SP;

    SBTObjectType(size_t varStructSize)
      : varStructSize(varStructSize)
    {}
    
    inline AbstractVariable::SP getAbstractVariable(const std::string &varName)
    {
      assert(variables.find(varName) != variables.end());
      return variables[varName];
    }

    void declareVariable(const std::string &varName,
                         OWLDataType type,
                         size_t offset);
    const size_t varStructSize;
    std::map<std::string,AbstractVariable::SP> variables;
  };

  void SBTObjectType::declareVariable(const std::string &varName,
                                      OWLDataType type,
                                      size_t offset)
  {
    switch (type) {
    case OWL_BUFFER_POINTER:
      variables[varName]
        = std::make_shared<BufferVariable>();
      break;
    case OWL_FLOAT3:
      variables[varName]
        = std::make_shared<AbstractVariableT<vec3f>>();
      break;
    default:
      PING; PRINT(type);
      OWL_NOTIMPLEMENTED;
    };
    // variables[varName] = std::make_shared<AbstractVariable>();
  }
    
  
  struct SBTObject : public Object
  {
    typedef std::shared_ptr<SBTObject> SP;

    SBTObject(SBTObjectType::SP objectType)
      : objectType(objectType)
    {}

    SBTObjectType::SP const objectType;
  };

  struct Buffer : public Object
  {
    typedef std::shared_ptr<Buffer> SP;
  };

  struct Geometry;
  
  struct GeometryType : public SBTObjectType {
    typedef std::shared_ptr<GeometryType> SP;
    
    GeometryType(size_t varStructSize)
      : SBTObjectType(varStructSize)
    {}

    virtual void setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
    { IGNORING_THIS(); }

    virtual std::shared_ptr<Geometry> createGeometry() = 0;
  };

  struct TrianglesGeometryType : public GeometryType {
    typedef std::shared_ptr<TrianglesGeometryType> SP;
    
    TrianglesGeometryType(size_t varStructSize)
      : GeometryType(varStructSize)
    {}

    virtual std::shared_ptr<Geometry> createGeometry() override;
  };

  struct UserGeometryType : public GeometryType {
    typedef std::shared_ptr<UserGeometryType> SP;
    
    UserGeometryType(size_t varStructSize)
      : GeometryType(varStructSize)
    {}

    virtual std::shared_ptr<Geometry> createGeometry() override;
  };

  struct Geometry : public SBTObject {
    typedef std::shared_ptr<Geometry> SP;

    Geometry(GeometryType::SP geometryType)
      : SBTObject(geometryType)
    {}
    
    GeometryType::SP geometryType;
  };

  struct TrianglesGeometry : public Geometry {
    typedef std::shared_ptr<TrianglesGeometry> SP;

    TrianglesGeometry(GeometryType::SP geometryType)
      : Geometry(geometryType)
    {}

    void setVertices(Buffer::SP vertices)
    { IGNORING_THIS(); }
    void setIndices(Buffer::SP indices)
    { IGNORING_THIS(); }
  };

  struct UserGeometry : public Geometry {
    typedef std::shared_ptr<UserGeometry> SP;

    UserGeometry(GeometryType::SP geometryType)
      : Geometry(geometryType)
    {}
  };
  
  // ==================================================================
  // apihandle.h
  // ==================================================================
  struct Context;
  
  struct APIHandle;

  struct APIHandle {
    APIHandle(Object::SP object, Context *context);
    virtual ~APIHandle();
    template<typename T> inline std::shared_ptr<T> get();
    inline std::shared_ptr<Context> getContext() const { return context; }
    inline bool isContext() const
    {
      // TODO: clean up with proper dynamic_cast ...
      return ((void*)object.get() == (void*)context.get());
    }
  private:
    std::shared_ptr<Object>     object;
    std::shared_ptr<Context>    context;
  };

  template<typename T> inline std::shared_ptr<T> APIHandle::get()
  {
    assert(object);
    std::shared_ptr<T> asT = std::dynamic_pointer_cast<T>(object);
    if (object && !asT)
      throw std::runtime_error("could not convert APIHandle of type "
                               + std::string(typeid(*object.get()).name())
                               + " to object of type "
                               + std::string(typeid(T).name()));
    assert(asT);
    return asT;
  }
    
  struct ApiHandles {
    void track(APIHandle *object)
    {
      assert(object);
      auto it = active.find(object);
      assert(it == active.end());
      active.insert(object);
    }
    
    void forget(APIHandle *object)
    {
      assert(object);
      auto it = active.find(object);
      assert(it != active.end());
      active.erase(it);
    }
    
    std::set<APIHandle *> active;
  };
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
  // END apihandle.h
  // <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<


  
  struct Context : public Object {
    typedef std::shared_ptr<Context> SP;
    
    virtual ~Context()
    {
      std::cout << "#owl: destroying context" << std::endl;
    }

    ApiHandles apiHandles;

    APIHandle *createHandle(Object::SP object)
    {
      assert(object);
      APIHandle *handle = new APIHandle(object,this);
      apiHandles.track(handle);
      return handle;
    }
    
    void releaseAll()
    {
      PING;
      std::cout << "#owl: context is dying, num api handles (other than context itself) that hvae not yet released: "
                << (apiHandles.active.size()-1)
                << std::endl;
      for (auto handle : apiHandles.active)
        delete handle;
    }

    Buffer::SP createBuffer();
    GeometryType::SP createGeometryType(OWLGeometryKind kind,
                                        size_t varStructSize);
    Module::SP createModule(const std::string &ptxCode);
    // Triangles::SP createTriangles(size_t varsStructSize);
  };

  APIHandle::APIHandle(Object::SP object, Context *context)
  {
    assert(object);
    assert(context);
    this->object  = object;
    this->context = std::dynamic_pointer_cast<Context>
      (context->shared_from_this());
    assert(this->object);
    assert(this->context);
  }

  APIHandle::~APIHandle()
  {
    context->apiHandles.forget(this);
    object  = nullptr;
    context = nullptr;
  }
  
  OWL_API OWLContext owlContextCreate()
  {
    LOG_API_CALL();
    Context::SP context = std::make_shared<Context>();
    std::cout << "#owl.api: context created..." << std::endl;
    return (OWLContext)context->createHandle(context);
  }

  OWL_API void owlContextLaunch2D(OWLContext context,
                                  OWLLaunchProg launchProg,
                                  int dims_x, int dims_y)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }


  OWL_API OWLVariable
  owlGeometryGetVariable(OWLGeometry geom,
                         const char *varName)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }

  OWL_API OWLVariable
  owlLaunchProgGetVariable(OWLLaunchProg geom,
                           const char *varName)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }
  


  OWL_API OWLLaunchProg
  owlContextCreateLaunchProg(OWLContext context,
                             OWLModule module,
                             const char *programName,
                             size_t sizeOfVarStruct)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }

  OWL_API OWLGeometryGroup
  owlContextCreateGeometryGroup(OWLContext context,
                                size_t numGeometries,
                                OWLGeometry *initValues)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }

  OWL_API OWLInstanceGroup
  owlContextCreateInstanceGroup(OWLContext context,
                                size_t numInstances)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }



  

  OWL_API void owlContextDestroy(OWLContext _context)
  {
    LOG_API_CALL();
    assert(_context);
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    assert(context);
    Buffer::SP  buffer  = context->createBuffer();
    assert(buffer);
    return (OWLBuffer)context->createHandle(buffer);
  }

  OWL_API OWLGeometryType
  owlContextCreateGeometryType(OWLContext _context,
                               OWLGeometryKind kind,
                               size_t varStructSize)
  {
    LOG_API_CALL();
    assert(_context);
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    assert(context);
    GeometryType::SP geometryType
      = context->createGeometryType(kind,varStructSize);
    assert(geometryType);
    return (OWLGeometryType)context->createHandle(geometryType);
  }

  OWL_API OWLGeometry
  owlContextCreateGeometry(OWLContext      _context,
                           OWLGeometryType _geometryType)
  {
    assert(_geometryType);
    assert(_context);

    Context::SP context
      = ((APIHandle *)_context)->get<Context>();
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
    
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    assert(context);
    Module::SP  module  = context->createModule(ptxCode);
    assert(module);
    return (OWLModule)context->createHandle(module);
  }


  
  Buffer::SP Context::createBuffer()
  {
    return std::make_shared<Buffer>();
  }

  GeometryType::SP Context::createGeometryType(OWLGeometryKind kind,
                                               size_t varStructSize)
  {
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      return std::make_shared<TrianglesGeometryType>(varStructSize);
    case OWL_GEOMETRY_USER:
      return std::make_shared<UserGeometryType>(varStructSize);
    default:
      OWL_NOTIMPLEMENTED;
    }
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    return std::make_shared<Module>(ptxCode);
  }

  std::shared_ptr<Geometry> UserGeometryType::createGeometry()
  {
    GeometryType::SP self
      = std::dynamic_pointer_cast<GeometryType>(shared_from_this());
    assert(self);
    return std::make_shared<UserGeometry>(self);
  }

  std::shared_ptr<Geometry> TrianglesGeometryType::createGeometry()
  {
    GeometryType::SP self
      = std::dynamic_pointer_cast<GeometryType>(shared_from_this());
    assert(self);
    return std::make_shared<TrianglesGeometry>(self);
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
  
  OWL_API void owlVariableRelease(OWLVariable variable)
  {
    LOG_API_CALL();
    releaseObject<BoundVariable>((APIHandle*)variable);
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
  // "VariableDeclare" functions, for each element type
  // ==================================================================

  template<typename T>
  void declareVariable(APIHandle  *handle,
                       const char *varName,
                       OWLDataType type,
                       size_t      offset)
  {
    assert(handle);
    assert(varName);
    
    typename T::SP object = handle->get<T>();
    assert(object);

    object->declareVariable(varName,type,offset);
  }

  OWL_API void
  owlGeometryTypeDeclareVariable(OWLGeometryType object,
                                 const char *varName,
                                 OWLDataType type,
                                 size_t offset)
  {
    LOG_API_CALL();
    declareVariable<GeometryType>
      ((APIHandle *)object,varName,type,offset);
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

  





  
  // OWL_API void owlGeometryTypeDeclareVariable(OWLGeometryType object,
  //                                             const char  *varName,
  //                                             OWLDataType  type,
  //                                             size_t       offset)
  // { declareVariable<Triangles>((APIHandle *)object,varName,type,offset); }
  
  // ==================================================================
  // "VariableSet" functions, for each element type
  // ==================================================================

  template<typename T>
  void setBasicTypeVariable(APIHandle *handle, const T &value)
  {
    assert(handle);

    BoundVariable::SP variable
      = handle->get<BoundVariable>();
    assert(variable);

    variable->set(value);
  }

  
  OWL_API void owlVariableSet1f(OWLVariable _variable, const float value)
  {
    LOG_API_CALL();
    setBasicTypeVariable((APIHandle *)_variable,(float)value);
    // LOG_API_CALL();
    // assert(_variable);
    // assert(value);

    // BoundVariable::SP variable
    //   = ((APIHandle *)_variable)->get<BoundVariable>();
    // assert(variable);

    // variable->set(value);
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
  owlInstanceGroupSetChild(OWLInstanceGroup group,
                           int whichChild,
                           OWLGeometryGroup geometry)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }
  
}
