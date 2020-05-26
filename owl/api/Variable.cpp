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

  struct UserTypeVariable : public Variable
  {
    UserTypeVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl),
        data(/* actual size is 'type' - constant */varDecl->type - OWL_USER_TYPE_BEGIN)
    {}
    
    void setRaw(const void *ptr) override
    {
      memcpy(data.data(),ptr,data.size());
    }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      memcpy(sbtEntry,data.data(),data.size());
    }
    
    std::vector<uint8_t> data;
  };
    
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
      const void *value
        = buffer
        ? buffer->getPointer(deviceID)
        : nullptr;
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
    void set(const Group::SP &value) override
    {
      if (value && !std::dynamic_pointer_cast<InstanceGroup>(value))
        throw std::runtime_error("OWL currently supports only instance groups to be passed to traversal; if you do want to trace rays into a single User or Triangle group, please put them into a single 'dummy' instance with jsut this one child and a identity transform");
      this->group = value;
    }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      const OptixTraversableHandle value
        = group
        ? group->context->llo->groupGetTraversable(group->ID,deviceID)
        : 0;
      *(OptixTraversableHandle*)sbtEntry = value;
    }
    
    
    Group::SP group;
  };
  



  struct TextureVariable : public Variable {
    typedef std::shared_ptr<TextureVariable> SP;

    TextureVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Texture::SP &value) override
    {
      this->texture = value;
    }

    void writeToSBT(uint8_t *sbtEntry, int deviceID) const override
    {
      cudaTextureObject_t to = {};
      if (texture) {
        assert(texture->textureObjects.size() > deviceID);
        to = texture->textureObjects[deviceID];
      }
      *(cudaTextureObject_t*)sbtEntry = to;
    }
    
    Texture::SP texture;
  };
  



  Variable::SP Variable::createInstanceOf(const OWLVarDecl *decl)
  {
    assert(decl);
    assert(decl->name);
    if (decl->type >= OWL_USER_TYPE_BEGIN)
      return std::make_shared<UserTypeVariable>(decl);
    switch(decl->type) {
    case OWL_INT:
      return std::make_shared<VariableT<int32_t>>(decl);
    case OWL_INT2:
      return std::make_shared<VariableT<vec2i>>(decl);
    case OWL_INT3:
      return std::make_shared<VariableT<vec3i>>(decl);
    case OWL_INT4:
      return std::make_shared<VariableT<vec4i>>(decl);

    case OWL_UINT:
      return std::make_shared<VariableT<uint32_t>>(decl);
    case OWL_UINT2:
      return std::make_shared<VariableT<vec2ui>>(decl);
    case OWL_UINT3:
      return std::make_shared<VariableT<vec3ui>>(decl);
    case OWL_UINT4:
      return std::make_shared<VariableT<vec4ui>>(decl);

      
    case OWL_LONG:
      return std::make_shared<VariableT<int64_t>>(decl);
    case OWL_LONG2:
      return std::make_shared<VariableT<vec2l>>(decl);
    case OWL_LONG3:
      return std::make_shared<VariableT<vec3l>>(decl);
    case OWL_LONG4:
      return std::make_shared<VariableT<vec4l>>(decl);

    case OWL_ULONG:
      return std::make_shared<VariableT<uint64_t>>(decl);
    case OWL_ULONG2:
      return std::make_shared<VariableT<vec2ul>>(decl);
    case OWL_ULONG3:
      return std::make_shared<VariableT<vec3ul>>(decl);
    case OWL_ULONG4:
      return std::make_shared<VariableT<vec4ul>>(decl);


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
    case OWL_TEXTURE:
      return std::make_shared<TextureVariable>(decl);
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
