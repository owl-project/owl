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

#pragma once

#include "Object.h"

namespace owl {

  struct Buffer;
  struct Group;
  struct Texture;

  /*! "Variable"s are associated with objects, and hold user-supplied
      data of a given type. The purpose of this is to allow owl to
      internally populate device-side Shader Binding Table (SBT)
      entries based on host-side supplied parameters - ie, if sets a
      Group, Buffer, etc, parameter on the host, then we will
      automatically translate that to the respecitve device data (in
      these examples a OptiXTraversablaHandle, or device pointer) when
      we write it into the SBT.

      To add some type-safety into OWL we create, for each paramter
      that the user declares for an object, a matching (templated)
      variable type; if the user then tries to set a variable of a
      different type than declared we'll throw a 'mismatchingType'
      expception */
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;

    Variable(const OWLVarDecl *const varDecl)
      : varDecl(varDecl)
    { assert(varDecl); }
    
    virtual void set(const std::shared_ptr<Buffer>  &value) { mismatchingType(); }
    virtual void set(const std::shared_ptr<Group>   &value) { mismatchingType(); }
    virtual void set(const std::shared_ptr<Texture> &value) { mismatchingType(); }
    
    virtual void setRaw(const void *ptr)    { mismatchingType(); }

    virtual void set(const int32_t &value)  { mismatchingType(); }
    virtual void set(const vec2i &value)    { mismatchingType(); }
    virtual void set(const vec3i &value)    { mismatchingType(); }
    virtual void set(const vec4i &value)    { mismatchingType(); }
    
    virtual void set(const uint32_t &value) { mismatchingType(); }
    virtual void set(const vec2ui &value)   { mismatchingType(); }
    virtual void set(const vec3ui &value)   { mismatchingType(); }
    virtual void set(const vec4ui &value)   { mismatchingType(); }
    
    virtual void set(const int64_t &value)  { mismatchingType(); }
    virtual void set(const vec2l &value)    { mismatchingType(); }
    virtual void set(const vec3l &value)    { mismatchingType(); }
    virtual void set(const vec4l &value)    { mismatchingType(); }
    
    virtual void set(const uint64_t &value) { mismatchingType(); }
    virtual void set(const vec2ul &value)   { mismatchingType(); }
    virtual void set(const vec3ul &value)   { mismatchingType(); }
    virtual void set(const vec4ul &value)   { mismatchingType(); }
    
    virtual void set(const float  &value)    { mismatchingType(); }
    virtual void set(const vec2f  &value)    { mismatchingType(); }
    virtual void set(const vec3f  &value)    { mismatchingType(); }
    virtual void set(const vec4f  &value)    { mismatchingType(); }

    virtual void set(const double &value)    { mismatchingType(); }
    virtual void set(const vec2d  &value)    { mismatchingType(); }
    virtual void set(const vec3d  &value)    { mismatchingType(); }
    virtual void set(const vec4d  &value)    { mismatchingType(); }
    
    virtual std::string toString() const { return "Variable"; }

    /*! throw an exception that the type the user tried to set doesn't
        math the type he/she declared*/
    void mismatchingType();

    /*! writes the device specific representation of the given type */
    virtual void writeToSBT(uint8_t *sbtEntry, const DeviceContext::SP &device) const = 0;

    /*! creates an instance of this variable type to be attached to a
        given object - this instance will can then store the values
        that the user passes */
    static Variable::SP createInstanceOf(const OWLVarDecl *decl);
    
    /*! the variable we're setting in the given object */
    const OWLVarDecl *const varDecl;
  };
  
} // ::owl
