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

#include "optix/Optix.h"

namespace optix {

  struct Context;

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

  
  struct Device {
    typedef std::shared_ptr<Device> SP;

    void setActive();
    
    std::mutex mutex;
    
    int                         cudaDeviceID;
    CUcontext                   cudaContext;
    CUstream                    stream;
    
    OptixDeviceContext          optixContext;
  };

  struct CommonBase {
    virtual std::string toString() = 0;
  };
  
  struct ObjectType : public CommonBase {
    typedef std::shared_ptr<ObjectType> SP;
    struct VariableSlot {
      size_t offset;
      size_t size;
    };
    std::map<std::string,VariableSlot> variableSlots;
  };

  struct Module : public CommonBase
  {
    typedef std::shared_ptr<Module> SP;
    std::string ptxCode;
  };
  
  struct Program : public CommonBase
  {
    typedef std::shared_ptr<Program> SP;
    
    Module::SP  module;
    std::string programName;
  };
  
  struct GeometryType : public ObjectType {
    typedef std::shared_ptr<GeometryType> SP;
    
    struct Programs {
      Program::SP intersect;
      Program::SP bounds;
      Program::SP anyHit;
      Program::SP closestHit;
    };
    //! one group of programs per ray type
    std::vector<Programs> programs;
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

    /*! creates a new context with one or more GPUs as specified in
        the selection method */
    static Context::SP create(GPUSelectionMethod whichGPUs=GPU_SELECT_FIRST_PREFER_RTX);
    
    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown */
    static Context::SP create(const std::vector<int> &deviceIDs);
    
    GeometryObject::SP createGeometryObject(GeometryType::SP type, size_t numPrims);
    
    std::mutex mutex;
    std::vector<Device::SP> allDevices;
    std::vector<Device::SP> activeDevices;
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
