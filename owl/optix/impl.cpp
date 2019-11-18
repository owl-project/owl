#include "owl/owl.h"
#include "optix/common.h"
#include <set>
#include <map>
#include <typeinfo>

namespace owl {
  using gdt::vec3f;
  
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;
    virtual ~Object() {}
  };
  
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;
    void wrongType() { PING; }
    
    virtual void set(const vec3f &value) { wrongType(); }
  };

  template<typename T>
  struct VariableT : public Object {
    typedef std::shared_ptr<VariableT<T>> SP;
    
    void set(const T &value) override { this->value = value; PING; }
    T value;
  };
  
  struct SBTObject : public Object
  {
    typedef std::shared_ptr<SBTObject> SP;

    SBTObject(size_t varStructSize)
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
    
    size_t varStructSize;
    std::map<std::string,Variable::SP> variables;
  };

  struct Buffer : public Object
  {
    typedef std::shared_ptr<Buffer> SP;
  };

  struct Geometry : public SBTObject {
    Geometry(size_t varStructSize) : SBTObject(varStructSize) {}
    
    typedef std::shared_ptr<Geometry> SP;
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
    // Triangles::SP createTriangles(size_t varsStructSize);
  };

  APIHandle::APIHandle(Object::SP object, Context *context)
  {
    PING;
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
    PING;
#if 1
    Context::SP context = std::make_shared<Context>();
    return (OWLContext)context->createHandle(context);
#else
    Context *context = nullptr;
    return (OWLContext)context;
#endif
  }

  OWL_API void owlContextDestroy(OWLContext _context)
  {
    PING;
    assert(_context);
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    // Context *context = (Context *)_context;
    context->releaseAll();
    // delete _context;
  }

  OWL_API OWLBuffer owlBufferCreate(OWLContext _context,
                                    OWLDataType type,
                                    int num,
                                    const void *init)
  {
    PING;
    assert(_context);
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    assert(context);
    Buffer::SP  buffer  = context->createBuffer();
    assert(buffer);
    return (OWLBuffer)context->createHandle(buffer);
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
  // Triangles::SP Context::createTriangles(size_t varsStructSize)
  // {
  //   return std::make_shared<Triangles>(varsStructSize);
  // }
  

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
  { releaseObject<Buffer>((APIHandle*)buffer); }
  
  OWL_API void owlVariableRelease(OWLVariable variable)
  { releaseObject<Variable>((APIHandle*)variable); }
  
  OWL_API void owlGeometryRelease(OWLGeometry geometry)
  { releaseObject<Geometry>((APIHandle*)geometry); }

  // // ==================================================================
  // // "Triangles" functions
  // // ==================================================================
  // OWL_API void owlTrianglesSetVertices(OWLTriangles _triangles,
  //                                     OWLBuffer    _vertices)
  // {
  //   Triangles::SP triangles = ((APIHandle *)_triangles)->get<Triangles>();
  //   Buffer::SP    vertices   = ((APIHandle *)_vertices)->get<Buffer>();
  //   triangles->vertices = vertices;
  // }

  // OWL_API void owlTrianglesSetIndices(OWLTriangles _triangles,
  //                                     OWLBuffer    _indices)
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
  
  // OWL_API void owlGeometryTypeDeclareVariable(OWLGeometryType object,
  //                                             const char  *varName,
  //                                             OWLDataType  type,
  //                                             size_t       offset)
  // { declareVariable<Triangles>((APIHandle *)object,varName,type,offset); }
  
  // ==================================================================
  // "VariableSet" functions, for each element type
  // ==================================================================

  OWL_API void owlVariableSet3fv(OWLVariable _variable, const float *value)
  {
    assert(_variable);
    assert(value);

    Variable::SP variable = ((APIHandle *)_variable)->get<Variable>();
    assert(variable);
    variable->set(*(const vec3f*)value);
  }
  
}
