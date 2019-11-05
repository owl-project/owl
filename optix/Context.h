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
    typedef std::shared_ptr<Context> SP;

    GeometryObject::SP createGeometryObject(GeometryType::SP type, size_t numPrims);
    
    std::mutex mutex;
    std::vector<Device::SP> devices;
  };
  
} // ::optix
