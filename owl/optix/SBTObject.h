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

#include "RegisteredObject.h"
#include "Variable.h"

namespace owl {

  struct SBTObjectType : public RegisteredObject
  {
    typedef std::shared_ptr<SBTObjectType> SP;

    SBTObjectType(Context *const context,
                  ObjectRegistry &registry,
                  size_t varStructSize,
                  const std::vector<OWLVarDecl> &varDecls)
      : RegisteredObject(context,registry),
        varStructSize(varStructSize),
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
  struct SBTObject : public RegisteredObject
  {
    typedef std::shared_ptr<SBTObject> SP;

    SBTObject(Context *const context,
              ObjectRegistry &registry,
              typename ObjectType::SP const type)
      : RegisteredObject(context,registry),
        type(type),
        variables(type->instantiateVariables())
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

} // ::owl

