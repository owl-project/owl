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
  
  struct Variable : public Object {
    typedef std::shared_ptr<Variable> SP;

    Variable(const OWLVarDecl *const varDecl)
      : varDecl(varDecl)
    { assert(varDecl); }
    
    virtual void set(const std::shared_ptr<Buffer> &value) { mismatchingType(); }
    virtual void set(const float &value) { mismatchingType(); }
    virtual void set(const vec3f &value) { mismatchingType(); }

    virtual std::string toString() const { return "Variable"; }
    
    void mismatchingType() { throw std::runtime_error("trying to set variable to value of wrong type"); }

    static Variable::SP createInstanceOf(const OWLVarDecl *decl);
    
    /*! the variable we're setting in the given object */
    const OWLVarDecl *const varDecl;
  };
  
} // ::owl

