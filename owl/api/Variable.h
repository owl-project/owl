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

#include "Object.h"

namespace owl {

  struct Buffer;
  struct Group;
  struct Texture;
  
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;

    Variable(const OWLVarDecl *const varDecl)
      : varDecl(varDecl)
    { assert(varDecl); }
    
    virtual void set(const std::shared_ptr<Buffer> &value) { mismatchingType(); }
    virtual void set(const std::shared_ptr<Group>  &value) { mismatchingType(); }
    virtual void set(const std::shared_ptr<Texture>  &value) { mismatchingType(); }
    
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
    
    virtual void set(const float &value)    { mismatchingType(); }
    virtual void set(const vec2f &value)    { mismatchingType(); }
    virtual void set(const vec3f &value)    { mismatchingType(); }
    virtual void set(const vec4f &value)    { mismatchingType(); }

    virtual std::string toString() const { return "Variable"; }
    
    void mismatchingType() { throw std::runtime_error("trying to set variable to value of wrong type"); }

    virtual void writeToSBT(uint8_t *sbtEntry, int deviceID) const;
    
    static Variable::SP createInstanceOf(const OWLVarDecl *decl);
    
    /*! the variable we're setting in the given object */
    const OWLVarDecl *const varDecl;
  };
  
} // ::owl

