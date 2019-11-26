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
#include <string.h>
#include <set>
#include <map>
#include <vector>
#include <stack>
#include <typeinfo>
#include <mutex>

namespace owl {
  using gdt::vec3f;

  struct Context;
  struct Buffer;
  template<typename Object> struct ObjectRegistry;
  struct Geometry;
  struct GeometryGroup;
  struct LaunchProg;

#define LOG_API_CALL() std::cout << "% " << __FUNCTION__ << "(...)" << std::endl;

#define IGNORING_THIS() std::cout << "## ignoring " << __PRETTY_FUNCTION__ << std::endl;
  
  
  //#define OWL_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+" : not implemented")
#define OWL_NOTIMPLEMENTED std::cerr << (std::string(__PRETTY_FUNCTION__)+" : not implemented") << std::endl; exit(1);


  std::string typeToString(const OWLDataType type)
  {
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                             +": not yet implemented for type #"
                             +std::to_string((int)type));
  }
  
  struct Object : public std::enable_shared_from_this<Object> {
    typedef std::shared_ptr<Object> SP;

    virtual std::string toString() const { return "Object"; }
    
    template<typename T>
    inline std::shared_ptr<T> as() 
    { return std::dynamic_pointer_cast<T>(shared_from_this()); }
    
    virtual ~Object() {}
  };

  template<typename T> const T &assertNotNull(const T &ptr)
  { assert(ptr); return ptr; }
  
  struct Variable;
  
  // struct VarDecl : public Object {

  //   typedef std::shared_ptr<VarDecl> SP;

  //   virtual std::string toString() const { return "VarDecl"; }

  //   /*! create an actual instance of this variable */
  //   virtual std::shared_ptr<Variable> instantiate() = 0;
  // };

  
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;

    Variable(const OWLVarDecl *const varDecl)
      : varDecl(varDecl)
    { assert(varDecl); }
    
    virtual void set(const std::shared_ptr<Buffer> &value) { wrongType(); }
    virtual void set(const float &value) { wrongType(); }
    virtual void set(const vec3f &value) { wrongType(); }

    virtual std::string toString() const { return "Variable"; }
    
    void wrongType() { PING; }

    static Variable::SP createInstanceOf(const OWLVarDecl *decl);
    
    /*! the variable we're setting in the given object */
    const OWLVarDecl *const varDecl;
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
    
    virtual std::string toString() const { return "Module"; }
    
    const std::string ptxCode;
  };
  
  struct SBTObjectType : public Object
  {
    typedef std::shared_ptr<SBTObjectType> SP;

    SBTObjectType(size_t varStructSize,
                  const std::vector<OWLVarDecl> &varDecls)
      : varStructSize(varStructSize),
        varDecls(varDecls)
    {
      for (auto &var : varDecls)
        assert(var.name != nullptr);
      /* TODO: at least in debug mode, do some 'duplicate variable
         name' and 'overlap of variables' checks etc */
    }
    
    inline int getVariableIdx(const std::string &varName)
    {
      for (int i=0;i<varDecls.size();i++) {
        assert(varDecls[i].name);
        if (!strcmp(varName.c_str(),varDecls[i].name))
          return i;
      }
      return -1;
    }
    inline bool hasVariable(const std::string &varName)
    {
      return getVariableIdx(varName) >= 0;
    }

    virtual std::string toString() const { return "SBTObjectType"; }
    void declareVariable(const std::string &varName,
                         OWLDataType type,
                         size_t offset);

    std::vector<Variable::SP> instantiateVariables();
    
    /*! the total size of the variables struct */
    const size_t         varStructSize;

    /*! the high-level semantic description of variables in the
        variables struct */
    const std::vector<OWLVarDecl> varDecls;
  };

  template<typename ObjectType>
  struct SBTObject : public Object
  {
    typedef std::shared_ptr<SBTObject> SP;

    SBTObject(typename ObjectType::SP const type)
      : type(type),
        variables(assertNotNull(type)->instantiateVariables())
    {}
    
    virtual std::string toString() const { return "SBTObject<"+type->toString()+">"; }
    
    bool hasVariable(const std::string &name)
    {
      return type->hasVariable(name);
    }
    
    Variable::SP getVariable(const std::string &name)
    {
      int varID = type->getVariableIdx(name);
      assert(varID >= 0);
      assert(varID <  variables.size());
      Variable::SP var = variables[varID];
      assert(var);
      return var;
    }

    /*! our own type description, that tells us which variables (of
        which type, etc) we have */
    typename ObjectType::SP const type;

    /*! the actual variable *values* */
    const std::vector<Variable::SP> variables;
  };

  struct LaunchProgType : public SBTObjectType {
    typedef std::shared_ptr<LaunchProgType> SP;
    LaunchProgType(Module::SP module,
                   const std::string &programName,
                   size_t varStructSize,
                   const std::vector<OWLVarDecl> &varDecls)
      : SBTObjectType(varStructSize,varDecls),
        module(module),
        progName(progName)
    {}
    virtual std::string toString() const { return "LaunchProgType"; }
    
    Module::SP module;
    const std::string &progName;
  };
  
  struct LaunchProg : public SBTObject<LaunchProgType> {
    typedef std::shared_ptr<LaunchProg> SP;

    LaunchProg(LaunchProgType::SP type) 
      : SBTObject(type)
    {}
    virtual std::string toString() const { return "LaunchProg"; }
  };

  struct Buffer : public Object
  {
    typedef std::shared_ptr<Buffer> SP;
    
    Buffer(ObjectRegistry<Buffer> &buffers);
    ~Buffer();
    
    virtual std::string toString() const { return "Buffer"; }
    const int ID;
  private:
    ObjectRegistry<Buffer> &buffers;
  };

  /*! registry that tracks mapping between buffers and buffer
      IDs. Every buffer should have a valid ID, and should be tracked
      in this registry under this ID */
  template<typename Object>
  struct ObjectRegistry {
    inline size_t size() const { return objects.size(); }
    
    void forget(Object *object)
    {
      assert(object);
      
      std::lock_guard<std::mutex> lock(mutex);
      assert(object->ID >= 0);
      assert(object->ID < objects.size());
      assert(objects[object->ID] == object);
      objects[object->ID] = nullptr;
      
      previouslyReleasedIDs.push(object->ID);
    }
    
    void track(Object *object)
    {
      assert(object);
      std::lock_guard<std::mutex> lock(mutex);
      assert(object->ID >= 0);
      assert(object->ID < objects.size());
      assert(objects[object->ID] == nullptr);
      objects[object->ID] = object;
    }
    
    int allocID() {
      std::lock_guard<std::mutex> lock(mutex);
      if (previouslyReleasedIDs.empty()) {
        objects.push_back(nullptr);
        return objects.size()-1;
      } else {
        int reusedID = previouslyReleasedIDs.top();
        previouslyReleasedIDs.pop();
        return reusedID;
      }
    }
    inline Object *get(int ID)
    {
      std::lock_guard<std::mutex> lock(mutex);
      
      assert(ID >= 0);
      assert(ID < objects.size());
      assert(objects[ID]);
      return objects[ID];
    }

  private:
    /*! list of all tracked objects. note this are *NOT* shared-ptr's,
        else we'd never released objects because each object would
        always be owned by the registry */
    std::vector<Object *> objects;
    
    /*! list of IDs that have already been allocated before, and have
        since gotten freed, so can be re-used */
    std::stack<int> previouslyReleasedIDs;
    std::mutex mutex;
  };



  // /*! registry that tracks mapping between buffers and buffer
  //     IDs. Every buffer should have a valid ID, and should be tracked
  //     in this registry under this ID */
  // struct BufferRegistry {
  //   void forget(Buffer *buffer)
  //   {
  //     assert(buffer);
      
  //     std::lock_guard<std::mutex> lock(mutex);
  //     assert(buffer->ID >= 0);
  //     assert(buffer->ID < buffers.size());
  //     assert(buffers[buffer->ID] == buffer);
  //     buffers[buffer->ID] = nullptr;
      
  //     previouslyReleasedIDs.push(buffer->ID);
  //   }
    
  //   void track(Buffer *buffer)
  //   {
  //     assert(buffer);
  //     std::lock_guard<std::mutex> lock(mutex);
  //     assert(buffer->ID >= 0);
  //     assert(buffer->ID < buffers.size());
  //     assert(buffers[buffer->ID] == nullptr);
  //     buffers[buffer->ID] = buffer;
  //   }
    
  //   int allocID() {
  //     std::lock_guard<std::mutex> lock(mutex);
  //     if (previouslyReleasedIDs.empty()) {
  //       buffers.push_back(nullptr);
  //       return buffers.size()-1;
  //     } else {
  //       int reusedID = previouslyReleasedIDs.top();
  //       previouslyReleasedIDs.pop();
  //       return reusedID;
  //     }
  //   }
  //   inline Buffer *get(int ID)
  //   {
  //     std::lock_guard<std::mutex> lock(mutex);
      
  //     assert(ID >= 0);
  //     assert(ID < buffers.size());
  //     assert(buffers[ID]);
  //     return buffers[ID];
  //   }

  // private:
  //   /*! list of all tracked buffers. note this are *NOT* shared-ptr's,
  //       else we'd never released buffers because each buffer would
  //       always be owned by the registry */
  //   std::vector<Buffer *> buffers;
    
  //   /*! list of IDs that have already been allocated before, and have
  //       since gotten freed, so can be re-used */
  //   std::stack<int> previouslyReleasedIDs;
  //   std::mutex mutex;
  // };



  Buffer::Buffer(ObjectRegistry<Buffer> &buffers)
    : ID(buffers.allocID()),
      buffers(buffers)
  {
    buffers.track(this);
  }
  
  Buffer::~Buffer()
  { buffers.forget(this); }
  

  struct GeometryType : public SBTObjectType {
    typedef std::shared_ptr<GeometryType> SP;
    
    GeometryType(size_t varStructSize,
                 const std::vector<OWLVarDecl> &varDecls)
      : SBTObjectType(varStructSize,varDecls)
    {}

    virtual std::string toString() const { return "GeometryType"; }
    virtual void setClosestHitProgram(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
    { IGNORING_THIS(); }

    virtual std::shared_ptr<Geometry> createGeometry() = 0;
  };

  struct TrianglesGeometryType : public GeometryType {
    typedef std::shared_ptr<TrianglesGeometryType> SP;
    
    TrianglesGeometryType(size_t varStructSize,
                          const std::vector<OWLVarDecl> &varDecls)
      : GeometryType(varStructSize,varDecls)
    {}

    virtual std::string toString() const { return "TrianlgeGeometryType"; }
    virtual std::shared_ptr<Geometry> createGeometry() override;
  };

  struct UserGeometryType : public GeometryType {
    typedef std::shared_ptr<UserGeometryType> SP;
    
    UserGeometryType(size_t varStructSize,
                 const std::vector<OWLVarDecl> &varDecls)
      : GeometryType(varStructSize,varDecls)
    {}

    virtual std::string toString() const { return "UserGeometryType"; }
    virtual std::shared_ptr<Geometry> createGeometry() override;
  };

  struct Geometry : public SBTObject<GeometryType> {
    typedef std::shared_ptr<Geometry> SP;

    Geometry(GeometryType::SP geometryType)
      : SBTObject(geometryType)
    {}
    virtual std::string toString() const { return "Geometry"; }
    
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
    virtual std::string toString() const { return "TrianglesGeometry"; }
  };

  struct UserGeometry : public Geometry {
    typedef std::shared_ptr<UserGeometry> SP;

    UserGeometry(GeometryType::SP geometryType)
      : Geometry(geometryType)
    {}
    virtual std::string toString() const { return "UserGeometry"; }
  };
  
  struct Group : public Object {
    Group(ObjectRegistry<Group> &groups);
    ~Group();
    virtual std::string toString() const { return "Group"; }
    
    const int ID;
  private:
    ObjectRegistry<Group> &groups;
  };


  Group::Group(ObjectRegistry<Group> &groups)
    : ID(groups.allocID()),
      groups(groups)
  {
    groups.track(this);
  }
  
  Group::~Group()
  { groups.forget(this); }
  
  
  struct GeometryGroup : public Group {
    typedef std::shared_ptr<GeometryGroup> SP;

    GeometryGroup(ObjectRegistry<Group> &groups, size_t numChildren)
      : Group(groups), children(numChildren)
    {}
    void setChild(int childID, Geometry::SP child)
    {
      assert(childID >= 0);
      assert(childID < children.size());
      children[childID] = child;
    }
    virtual std::string toString() const { return "GeometryGroup"; }
    std::vector<Geometry::SP> children;
  };

  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;

    InstanceGroup(ObjectRegistry<Group> &groups, size_t numChildren)
      : Group(groups), children(numChildren)
    {}
    void setChild(int childID, Group::SP child)
    {
      assert(childID >= 0);
      assert(childID < children.size());
      children[childID] = child;
    }
    virtual std::string toString() const { return "InstanceGroup"; }
    std::vector<Group::SP> children;
  };

  // ==================================================================
  // apihandle.h
  // ==================================================================
  
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
  // private:
    std::shared_ptr<Object>     object;
    std::shared_ptr<Context>    context;
  };

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
    
  struct ApiHandles {
    ~ApiHandles()
    {
    }
    
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

    /*! delete - and thereby, release - all handles that we still
        own. */
    void releaseAll()
    {
      // create a COPY of the handles we need to destroy, else
      // destroying the handles modifies the std::set while we're
      // iterating through it!
      std::set<APIHandle *> stillActiveHandles = active;
      for (auto handle : stillActiveHandles)  {
        assert(handle);
        delete handle;
      }

      assert(active.empty());
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
      std::cout << "=======================================================" << std::endl;
      std::cout << "#owl: destroying context" << std::endl;
    }

    ApiHandles apiHandles;
    ObjectRegistry<Buffer> buffers;
    ObjectRegistry<Group>  groups;

    /*! experimentation code for sbt construction */
    void expBuildSBT();
    
    APIHandle *createHandle(Object::SP object)
    {
      assert(object);
      APIHandle *handle = new APIHandle(object,this);
      apiHandles.track(handle);
      return handle;
    }
    
    void releaseAll()
    {
      std::cout << "#owl: context is dying, num api handles (other than context itself) "
                << "that have not yet released: "
                << (apiHandles.active.size()-1)
                << std::endl;
      apiHandles.releaseAll();
    }

    InstanceGroup::SP createInstanceGroup(size_t numChildren);
    GeometryGroup::SP createGeometryGroup(size_t numChildren);
    Buffer::SP        createBuffer();
    
    LaunchProg::SP
    createLaunchProg(const std::shared_ptr<LaunchProgType> &type);
    
    LaunchProgType::SP
    createLaunchProgType(Module::SP module,
                         const std::string &progName,
                         size_t varStructSize,
                         const std::vector<OWLVarDecl> &varDecls);
    
    GeometryType::SP
    createGeometryType(OWLGeometryKind kind,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);
    
    Module::SP createModule(const std::string &ptxCode);
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

  OWL_API void owlContextLaunch2D(OWLContext _context,
                                  OWLLaunchProg _launchProg,
                                  int dims_x, int dims_y)
  {
    LOG_API_CALL();

    assert(_context);
    Context::SP context
      = ((APIHandle *)_context)->get<Context>();
    assert(context);

    std::cout << "SHOULD BUILD SBT HERE!!!!" << std::endl;
    context->expBuildSBT();
    
    std::cout << "actual laumch not yet implemented - ignoring ...." << std::endl;
  }


  void Context::expBuildSBT()
  {
    PING;
    PRINT(groups.size());
    PING;
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

    Context::SP context = handle->getContext();
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
  owlLaunchProgGetVariable(OWLLaunchProg _prog,
                           const char *varName)
  {
    LOG_API_CALL();
    return getVariableHelper<LaunchProg>((APIHandle*)_prog,varName);
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

  OWL_API OWLLaunchProg
  owlContextCreateLaunchProg(OWLContext _context,
                             OWLModule _module,
                             const char *programName,
                             size_t sizeOfVarStruct,
                             OWLVarDecl *vars,
                             size_t      numVars)
  {
    LOG_API_CALL();

    assert(_context);
    Context::SP context
      = ((APIHandle *)_context)->get<Context>();
    assert(context);
    
    assert(_module);
    Module::SP module
      = ((APIHandle *)_module)->get<Module>();
    assert(module);

    LaunchProgType::SP  launchProgType
      = context->createLaunchProgType(module,programName,
                                      sizeOfVarStruct,
                                      checkAndPackVariables(vars,numVars));
    assert(launchProgType);
    
    LaunchProg::SP  launchProg
      = context->createLaunchProg(launchProgType);
    assert(launchProg);

    return (OWLLaunchProg)context->createHandle(launchProg);
  }

  OWL_API OWLGeometryGroup
  owlContextCreateGeometryGroup(OWLContext _context,
                                size_t numGeometries,
                                OWLGeometry *initValues)
  {
    LOG_API_CALL();
    assert(_context);
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    Context::SP context = ((APIHandle *)_context)->get<Context>();
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
    PING;
    Buffer::SP buffer = std::make_shared<Buffer>(buffers);
    PING;
    assert(buffer);
    return buffer;
  }

  LaunchProg::SP
  Context::createLaunchProg(const std::shared_ptr<LaunchProgType> &type)
  {
    return std::make_shared<LaunchProg>(type);
  }

  GeometryGroup::SP Context::createGeometryGroup(size_t numChildren)
  {
    return std::make_shared<GeometryGroup>(groups,numChildren);
  }

  InstanceGroup::SP Context::createInstanceGroup(size_t numChildren)
  {
    return std::make_shared<InstanceGroup>(groups,numChildren);
  }


  LaunchProgType::SP
  Context::createLaunchProgType(Module::SP module,
                                const std::string &progName,
                                size_t varStructSize,
                                const std::vector<OWLVarDecl> &varDecls)
  {
    return std::make_shared<LaunchProgType>(module,progName,
                                            varStructSize,
                                            varDecls);
  }
  

  GeometryType::SP
  Context::createGeometryType(OWLGeometryKind kind,
                              size_t varStructSize,
                              const std::vector<OWLVarDecl> &varDecls)
  {
    switch(kind) {
    case OWL_GEOMETRY_TRIANGLES:
      return std::make_shared<TrianglesGeometryType>(varStructSize,varDecls);
    case OWL_GEOMETRY_USER:
      return std::make_shared<UserGeometryType>(varStructSize,varDecls);
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
