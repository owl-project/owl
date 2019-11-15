#include "c-api.h"
#include "optix/common.h"
#include <set>
#include <map>

namespace owl {

  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;
    virtual ~Object() {}
  };
  
  struct Buffer : public Object
  {
    typedef std::shared_ptr<Buffer> SP;
  };

  struct Triangles : public Object {
    typedef std::shared_ptr<Triangles> SP;
    
    std::shared_ptr<Buffer> vertices;
    std::shared_ptr<Buffer> indices;
  };
  
  struct Context;

  struct APIHandle;

  struct APIHandle {
    APIHandle(Object::SP object, Context *context);

    std::shared_ptr<Object>     object;
    std::shared_ptr<Context>    context;

    template<typename T>
    inline std::shared_ptr<T> get()
    { assert(object); return std::dynamic_pointer_cast<T>(object); }
    
    virtual ~APIHandle();
  };

  
  // struct APIHandle {
  //   virtual void clear() = 0;
  //   virtual ~APIHandle() {}
  // };

  struct ApiHandles {
    void track(APIHandle *object)
    {
      auto it = active.find(object);
      assert(it == active.end());
      active.insert(object);
    }
    
    void forget(APIHandle *object)
    {
      auto it = active.find(object);
      assert(it != active.end());
      active.erase(it);
    }
    
    std::set<APIHandle *> active;
  };
  
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
      std::cout << "#owl: context is dying, num api handles not yet released: " << apiHandles.active.size() << std::endl;
      for (auto handle : apiHandles.active)
        delete handle;
    }

    Buffer::SP createBuffer();
    Triangles::SP createTriangles();
  };

  APIHandle::APIHandle(Object::SP object, Context *context)
  {
    this->object = object;
    this->context = std::dynamic_pointer_cast<Context>
      (context->shared_from_this());
  }

  APIHandle::~APIHandle()
  {
    context->apiHandles.forget(this);
    object  = nullptr;
    context = nullptr;
  }
  
  OWL_API OWLContext owlContextCreate()
  {
    Context *context = nullptr;
    return (OWLContext)context;
  }

  OWL_API void owlContextDestroy(OWLContext _context)
  {
    assert(_context);
    Context *context = (Context *)_context;
    context->releaseAll();
    delete context;
  }

  OWL_API OWLBuffer owlBufferCreate(OWLContext _context,
                                    OWLDataType type,
                                    int num,
                                    const void *init)
  {
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    return (OWLBuffer)context->createHandle(context->createBuffer());
  }

  OWL_API OWLTriangles owlTrianglesCreate(OWLContext _context)
  {
    Context::SP context = ((APIHandle *)_context)->get<Context>();
    return (OWLTriangles)context->createHandle(context->createTriangles());
  }

  Buffer::SP Context::createBuffer()
  {
    return std::make_shared<Buffer>();
  }
  Triangles::SP Context::createTriangles()
  {
    return std::make_shared<Triangles>();
  }
  

  // ==================================================================
  // "RELEASE" functions
  // ==================================================================
  OWL_API void owlObjectRelease(OWLObject objectHandle)
  {
    delete (APIHandle *)objectHandle;
  }

  OWL_API void owlBufferRelease(OWLBuffer buffer)
  {
    assert(buffer);
    owlObjectRelease((OWLObject)buffer);
  }
  
  OWL_API void owlVariableRelease(OWLVariable variable)
  {
    assert(variable);
    owlObjectRelease((OWLObject)variable);
  }
  
  OWL_API void owlTrianglesRelease(OWLTriangles triangles)
  {
    assert(triangles);
    owlObjectRelease((OWLObject)triangles);
  }

  // ==================================================================
  // "Triangles" functions
  // ==================================================================
  OWL_API void owlTrianglesSetVertices(OWLTriangles _triangles,
                                      OWLBuffer    _vertices)
  {
    Triangles::SP triangles = ((APIHandle *)_triangles)->get<Triangles>();
    Buffer::SP    vertices   = ((APIHandle *)_vertices)->get<Buffer>();
    triangles->vertices = vertices;
  }

  OWL_API void owlTrianglesSetIndices(OWLTriangles _triangles,
                                      OWLBuffer    _indices)
  {
    Triangles::SP triangles = ((APIHandle *)_triangles)->get<Triangles>();
    Buffer::SP    indices   = ((APIHandle *)_indices)->get<Buffer>();
    triangles->indices = indices;
  }

}
