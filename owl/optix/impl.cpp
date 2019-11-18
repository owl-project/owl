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

#define LOG_API_CALL() std::cout << "% " << __FUNCTION__ << "(...)" << std::endl;

#define IGNORING_THIS() std::cout << "## ignoring " << __PRETTY_FUNCTION__ << "(...)" << std::endl;
  
  
#define OWL_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" : not implemented")
  
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;
    virtual ~Object() {}
  };
  
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;
    void wrongType() { PING; }
    
    virtual void set(const float  value) { wrongType(); }
    virtual void set(const vec3f &value) { wrongType(); }
  };

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
  
  template<typename T>
  struct VariableT : public Object {
    typedef std::shared_ptr<VariableT<T>> SP;
    
    void set(const T &value) override { this->value = value; PING; }
    T value;
  };
  
  struct SBTObjectType : public Object
  {
    typedef std::shared_ptr<SBTObjectType> SP;

    SBTObjectType(size_t varStructSize)
      : varStructSize(varStructSize)
    {}
    
    inline Variable::SP getVariable(const std::string &varName)
    {
      assert(variables.find(varName) != variables.end());
      return variables[varName];
    }

    void declareVariable(const std::string &varName,
                         OWLDataType type,
                         size_t offset)
    {
      PING;
      variables[varName] = std::make_shared<Variable>();
    }
    
    const size_t varStructSize;
    std::map<std::string,Variable::SP> variables;
  };

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

    virtual std::shared_ptr<Geometry> createObject();
  };

  struct Geometry : public SBTObject {
    typedef std::shared_ptr<Geometry> SP;

    Geometry(GeometryType::SP geometryType)
      : SBTObject(geometryType)
    {}
    
    GeometryType::SP geometryType;
  };

  // struct Triangles : public SBTObject {
  //   Triangles(size_t varStructSize) : SBTObject(varStructSize) {}
    
  //   typedef std::shared_ptr<Triangles> SP;
    
  //   std::shared_ptr<Buffer> vertices;
  //   std::shared_ptr<Buffer> indices;
  // };

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
                               + std::string(typeid(T()).name()));
    assert(asT);
    return asT;
  }
    
  // struct APIHandle {
  //   virtual void clear() = 0;
  //   virtual ~APIHandle() {}
  // };

  struct ApiHandles {
    void track(APIHandle *object)
    {
      assert(object);
      // if (object->isContext())
      //   // contexts will NOT track themselves, else
      //   // 'Context::releaseAll()' will destroy the context itself
      //   return;
      auto it = active.find(object);
      assert(it == active.end());
      active.insert(object);
    }
    
    void forget(APIHandle *object)
    {
      assert(object);
      // if (object->isContext())
      //   // contexts will NOT track themselves, else
      //   // 'Context::releaseAll()' will destroy the context itself
      //   return;
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
    PING;
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
      = geometryType->createObject();
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


  
  // OWL_API OWLTriangles owlTrianglesCreate(OWLContext _context,
  //                                         size_t varsStructSize)
  // {
  //   assert(_context);
    
  //   Context::SP   context   = ((APIHandle *)_context)->get<Context>();
  //   assert(context);
    
  //   Triangles::SP triangles = context->createTriangles(varsStructSize);
  //   assert(triangles);
    
  //   APIHandle *handle       = context->createHandle(triangles);
  //   assert(handle);
    
  //   return (OWLTriangles)handle;
  // }

  Buffer::SP Context::createBuffer()
  {
    return std::make_shared<Buffer>();
  }

  GeometryType::SP Context::createGeometryType(OWLGeometryKind kind,
                                               size_t varStructSize)
  {
    return std::make_shared<GeometryType>(varStructSize);
  }

  Module::SP Context::createModule(const std::string &ptxCode)
  {
    return std::make_shared<Module>(ptxCode);
  }

  std::shared_ptr<Geometry> GeometryType::createObject()
  {
    GeometryType::SP self
      = std::dynamic_pointer_cast<GeometryType>(shared_from_this());
    assert(self);
    return std::make_shared<Geometry>(self);
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
                          OWLBuffer    _vertices)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }
  // {
  //   Triangles::SP triangles = ((APIHandle *)_triangles)->get<Triangles>();
  //   Buffer::SP    vertices   = ((APIHandle *)_vertices)->get<Buffer>();
  //   triangles->vertices = vertices;
  // }

  OWL_API void
  owlTrianglesSetIndices(OWLGeometry  _triangles,
                         OWLBuffer    _indices)
  {
    LOG_API_CALL();
    OWL_NOTIMPLEMENTED;
  }
  // {
  //   Triangles::SP triangles = ((APIHandle *)_triangles)->get<Triangles>();
  //   Buffer::SP    indices   = ((APIHandle *)_indices)->get<Buffer>();
  //   triangles->indices = indices;
  // }

  // ==================================================================
  // "GetVariable" functions, for each object type
  // ==================================================================

  // OWL_API OWLVariable owlGeometryGetVariable(OWLGeometry geometry,
  //                                            const char *varName)
  // { 
  //   SBTObject::SP object  = ((APIHandle *)_object)->get<SBTObject>();
  //   Context::SP   context = ((APIHandle *)_object)->getContext();
  //   return (OWLVariable)context->createHandle(object->getVariable(varName));
  // }
  
  // OWL_API OWLVariable owlTrianglesGetVariable(OWLTriangles _triangles,
  //                                             const char *varName)
  // { return (OWLVariable)owlGetVariable((OWLObject)_triangles,varName); }
  
  
  
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
    assert(_progName);

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

  OWL_API void owlVariableSet1f(OWLVariable _variable, const float value)
  {
    LOG_API_CALL();
    assert(_variable);
    assert(value);

    Variable::SP variable = ((APIHandle *)_variable)->get<Variable>();
    assert(variable);
    variable->set(value);
  }

  OWL_API void owlVariableSet3fv(OWLVariable _variable, const float *value)
  {
    LOG_API_CALL();
    assert(_variable);
    assert(value);

    Variable::SP variable = ((APIHandle *)_variable)->get<Variable>();
    assert(variable);
    variable->set(*(const vec3f*)value);
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
