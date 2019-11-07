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

#pragma once

#include "optix/Base.h"

namespace optix {

  using gdt::vec2i;
  
  struct Context;
  typedef std::shared_ptr<Context> ContextSP;

  struct Module;
  typedef std::shared_ptr<Module> ModuleSP;

  struct Program;
  typedef std::shared_ptr<Program> ProgramSP;

  /*! the basic abstraction for all classes owned by a optix
      context */
  struct Object {
    typedef std::shared_ptr<Object> SP;

    Object(std::weak_ptr<Context> context) : context(context) {}
    
    //! pretty-printer, for debugging
    virtual std::string toString() = 0;

    std::weak_ptr<Context> getContext() const { return context; }
  private:
    // the context owning this object
    std::weak_ptr<Context> context;
  };
  
  struct ObjectType : public CommonBase {
    typedef std::shared_ptr<ObjectType> SP;
    struct VariableSlot {
      size_t offset;
      size_t size;
    };
    std::map<std::string,VariableSlot> variableSlots;
  };

  struct GeometryType : public ObjectType {
    typedef std::shared_ptr<GeometryType> SP;
    
    struct Programs {
      ProgramSP intersect;
      ProgramSP bounds;
      ProgramSP anyHit;
      ProgramSP closestHit;
    };
    //! one group of programs per ray type
    std::vector<Programs> perRayType;
  };
  
  struct ParamObject : public CommonBase {
    ObjectType::SP type;
  };
  
  struct GeometryObject : public ParamObject {
    typedef std::shared_ptr<GeometryObject> SP;
  };

  
  /*! the root optix context that creates and managed all objects */
  struct Context {
    /*! used to specify which GPU(s) we want to use in a context */
    typedef enum
      {
       /*! take the first GPU, whichever one it is */
       GPU_SELECT_FIRST=0,
       /*! take the first RTX-enabled GPU, if available,
         else take the first you find - not yet implemented */
       GPU_SELECT_FIRST_PREFER_RTX,
       /*! leave it to owl to select which one to use ... */
       GPU_SELECT_BEST,
       /*! use *all* GPUs, in multi-gpu mode */
       GPU_SELECT_ALL,
       /*! use all RTX-enabled GPUs, in multi-gpu mode */
       GPU_SELECT_ALL_RTX
    } GPUSelectionMethod;
  
    typedef std::shared_ptr<Context> SP;
    typedef std::weak_ptr<Context>   WP;

    /*! creates a new context with one or more GPUs as specified in
        the selection method */
    static Context::SP create(GPUSelectionMethod whichGPUs=GPU_SELECT_FIRST);
    
    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown */
    static Context::SP create(const std::vector<uint32_t> &deviceIDs);

    /*! optix logging callback */
    static void log_cb(unsigned int level,
                       const char *tag,
                       const char *message,
                       void * /*cbdata */);
    
    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown. 

        will throw an error if no device(s) could be found for this context

        Should never be called directly, only through Context::create() */
    Context(const std::vector<uint32_t> &deviceIDs);
    
    GeometryObject::SP createGeometryObject(GeometryType::SP type,
                                            size_t numPrims);

    /*! create a new module object from given ptx string */
    ModuleSP  createModuleFromString(const std::string &ptxCode);

    std::vector<ProgramSP> entryPoints;
    
    /*! set raygen program name and module for given entry point */
    void setEntryPoint(size_t entryPointID,
                       ModuleSP module,
                       const std::string &programName);

    void launch(int entryPointID, const vec2i &size)
    { OWL_NOTIMPLEMENTED; }
    
    /*! a mutex for this particular context */
    std::mutex mutex;
    
    /*! mutex that's GLOBALLY present for all operations that do not
        have a context to operate on */
    static std::mutex        g_mutex;

    static void g_init();
    
    struct PerDevice {
      typedef std::shared_ptr<PerDevice> SP;
      typedef std::weak_ptr<PerDevice>   WP;

    private:
      friend class Context;

      PerDevice(Context *self,
                int cudaID,
                int optixID,
                CUcontext          cudaContext,
                CUstream           stream,
                OptixDeviceContext optixContext);

      /*! create the given optix device on given cuda device. will
          throw an error if for whatever reason this cannot be done */
      static PerDevice::SP create(int cudaDeviceID,
                                  int optixDeviceID,
                                  Context *self);

    public:
      std::mutex               mutex;
      Context                 *const self;
      
      /*! the ID of this device *AS SEEN BY CUDA* */
      const int                cudaID;
      
      /*! the ID of this device *WITHIN THE CONTEXT* */
      const int                optixID;

      /*! a cuda context for the given device */
      const CUcontext          cudaContext;
    
      /*! a stream we create for this cude context; you can of course
        use other streams as well, but this should be the default
        stream to be used for this device (in order to allow for this
        device to work independently of other devices */
      const CUstream           stream;

      /*! the (low-level!) optix device context for this device - NOT to
        be confused with the (high-level) optix::Context created by
        this library */
      const OptixDeviceContext optixContext;
    };

    size_t                      t_pipelineOptionsChanged = 0;
    
    /*! should only once be called by the constructor, to initialize
      all compile/link options to defaults */
    void initializePipelineDefaults();
    
    OptixModuleCompileOptions   moduleCompileOptions;
    OptixPipelineCompileOptions pipelineCompileOptions;
    OptixPipelineLinkOptions    pipelineLinkOptions;
    
    /*! list of all devices active in this context */
    std::vector<PerDevice::SP>  perDevice;
  };


  
  /*! base class for any kind of owl/optix exception that this lib
      could possibly throw */
  struct Error : public std::runtime_error {
    Error(const std::string &where,
              const std::string &what)
      : std::runtime_error(where+" : "+what)
    {}
  };
  
} // ::optix
