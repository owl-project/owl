// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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
#include "InstanceGroup.h"
// device buffer representation that we'll write for Buffer variables
#include "owl/owl_device_buffer.h"

namespace owl {
  
  /*! throw an exception that the type the user tried to set doesn't
    math the type he/she declared*/
  void Variable::mismatchingType()
  {
    throw std::runtime_error("trying to set variable to value of wrong type");
  }

  /*! Variable type for ray "user yypes". User types have a
      user-specified size in bytes, and get set by passing a pointer
      to 'raw' data that then gets copied in binary form */
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

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      memcpy(sbtEntry,data.data(),data.size());
    }
    
    std::vector<uint8_t> data;
  };

  /*! Variable type for basic and compound-basic data types such as
      float, vec3f, etc */
  template<typename T>
  struct VariableT : public Variable {
    typedef std::shared_ptr<VariableT<T>> SP;

    VariableT(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    
    void set(const T &value) override { this->value = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      *(T*)sbtEntry = value;
    }

    T value;
  };

  /*! Variable type that accepts owl buffer types, and on the
      device-side writes just the raw device pointer into the SBT */
  struct BufferPointerVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    BufferPointerVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const void *value
        = buffer
        ? buffer->getPointer(device)
        : nullptr;
      *(const void**)sbtEntry = value;
    }
    
    Buffer::SP buffer;
  };

  /*! Fully-implicit Variable type that doesn't actually take _any_
      user data, but instead always writes the currently active
      device's device ID into the SBT */
  struct DeviceIndexVariable : public Variable {
    typedef std::shared_ptr<BufferPointerVariable> SP;

    DeviceIndexVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override
    {
      throw std::runtime_error("cannot _set_ a device index variable; it is purely implicit");
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      *(int*)sbtEntry = device->ID;
    }
    
    Buffer::SP buffer;
  };
  
  /*! Buffer variable that takes owl buffer types, and on the device
      writes full device::Buffer types with size, pointer, etc */
  struct BufferVariable : public Variable {
    typedef std::shared_ptr<BufferVariable> SP;

    BufferVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Buffer::SP &value) override { this->buffer = value; }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      device::Buffer *devRep = (device::Buffer *)sbtEntry;
      if (!buffer) {
        devRep->data  = 0;
        devRep->count = 0;
        devRep->type  = OWL_INVALID_TYPE;
      } else {
        devRep->data  = (void *)buffer->getPointer(device);
        devRep->count = buffer->elementCount;
        devRep->type  = buffer->type;
      }
    }
    
    Buffer::SP buffer;
  };

  /*! Variable type that accepts owl Group types on the host, and
      writes the groups' respective OptixTraversableHandle into the
      SBT */
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

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      const OptixTraversableHandle value
        = group
        ? group->getTraversable(device)//context->llo->groupGetTraversable(group->ID,deviceID)
        : 0;
      *(OptixTraversableHandle*)sbtEntry = value;
    }
    
    
    Group::SP group;
  };
  
  /*! Variable type that manages textures; accepting owl::Texture
      objects on the host, and writing their corresponding cuda
      texture obejct handles into the SBT */
  struct TextureVariable : public Variable {
    typedef std::shared_ptr<TextureVariable> SP;

    TextureVariable(const OWLVarDecl *const varDecl)
      : Variable(varDecl)
    {}
    void set(const Texture::SP &value) override
    {
      this->texture = value;
    }

    /*! writes the device specific representation of the given type */
    void writeToSBT(uint8_t *sbtEntry,
                    const DeviceContext::SP &device) const override
    {
      cudaTextureObject_t to = {};
      if (texture) {
        assert(device->ID < texture->textureObjects.size());
        to = texture->textureObjects[device->ID];
      }
      *(cudaTextureObject_t*)sbtEntry = to;
    }
    
    Texture::SP texture;
  };
  
  /*! creates a variable type that matches the given variable
      declaration */
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

    case OWL_DOUBLE:
      return std::make_shared<VariableT<double>>(decl);
    case OWL_DOUBLE2:
      return std::make_shared<VariableT<vec2d>>(decl);
    case OWL_DOUBLE3:
      return std::make_shared<VariableT<vec3d>>(decl);
    case OWL_DOUBLE4:
      return std::make_shared<VariableT<vec4d>>(decl);

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
