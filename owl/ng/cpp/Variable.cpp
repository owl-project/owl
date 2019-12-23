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

#include "Variable.h"
#include "Context.h"

namespace owl {
  
  void Variable::writeToSBT(uint8_t *sbtEntry, int deviceID) const
  {
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                             +": not yet implemented for type "
                             +typeToString(varDecl->type));
  }

  template<typename T>
  struct VariableT : public Variable {
    typedef std::shared_ptr<VariableT<T>> SP;

    VariableT(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    
    void set(const T &value) override { this->value = value; }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      *(T*)sbtEntry = value;
    }

    T value;
  };

  struct BufferPointerVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    BufferPointerVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      assert(buffer);
      const void *value = buffer->getPointer(deviceID);
      *(const void**)sbtEntry = value;
    }
    
    Buffer::SP buffer;
  };
  
  struct DeviceIndexVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    DeviceIndexVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override
    {
      throw std::runtime_error("cannot _set_ a device index variable; it is purely implicit");
    }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      *(int*)sbtEntry = deviceID;
    }
    
    Buffer::SP buffer;
  };
  
  struct BufferVariable : public Variable {
    typedef std::shared_ptr<BufferVariable> SP;

    BufferVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    Buffer::SP buffer;
  };
  
  struct GroupVariable : public Variable {
    typedef std::shared_ptr<GroupVariable> SP;

    GroupVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Group::SP &value) override { this->group = value; }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      const OptixTraversableHandle value
        = lloGroupGetTraversable(group->context->llo,group->ID,deviceID);
      *(OptixTraversableHandle*)sbtEntry = value;
    }
    
    
    Group::SP group;
  };
  
  Variable::SP Variable::createInstanceOf(const OWLVarDecl *decl)
  {
    assert(decl);
    assert(decl->name);
    switch(decl->type) {
    case OWL_INT:
      return std::make_shared<VariableT<int>>(decl);
    case OWL_INT2:
      return std::make_shared<VariableT<vec2i>>(decl);
    case OWL_INT3:
      return std::make_shared<VariableT<vec3i>>(decl);
    case OWL_INT4:
      return std::make_shared<VariableT<vec4i>>(decl);

    case OWL_UINT:
      return std::make_shared<VariableT<int>>(decl);
    case OWL_UINT2:
      return std::make_shared<VariableT<vec2ui>>(decl);
    case OWL_UINT3:
      return std::make_shared<VariableT<vec3ui>>(decl);
    case OWL_UINT4:
      return std::make_shared<VariableT<vec4ui>>(decl);

    case OWL_FLOAT:
      return std::make_shared<VariableT<float>>(decl);
    case OWL_FLOAT2:
      return std::make_shared<VariableT<vec2f>>(decl);
    case OWL_FLOAT3:
      return std::make_shared<VariableT<vec3f>>(decl);
    case OWL_FLOAT4:
      return std::make_shared<VariableT<vec4f>>(decl);

    case OWL_GROUP:
      return std::make_shared<GroupVariable>(decl);
    case OWL_BUFFER:
      return std::make_shared<BufferVariable>(decl);
    case OWL_BUFFER_POINTER:
      return std::make_shared<BufferPointerVariable>(decl);
    case OWL_DEVICE:
      return std::make_shared<DeviceIndexVariable>(decl);
    }
    throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                             +": not yet implemented for type "
                             +typeToString(decl->type));
  }
    
} // ::owl
