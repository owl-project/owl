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
                  const std::vector<OWLVarDecl> &varDecls);
    
    int getVariableIdx(const std::string &varName);
    bool hasVariable(const std::string &varName);

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





  struct SBTObjectBase : public RegisteredObject
  {
    SBTObjectBase(Context *const context,
                  ObjectRegistry &registry,
                  std::shared_ptr<SBTObjectType> type)
      : RegisteredObject(context,registry),
        type(type),
        variables(type->instantiateVariables())
    {
    }

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

    /*! this function is arguably the heart of the NG layer: given an
      SBT Object's set of variables, create the SBT entry that writes
      the given variables' values into the specified format, prorperly
      translating per-device data (buffers, traversable) while doing
      so */
    void writeVariables(uint8_t *sbtEntry,
                        int deviceID) const;
    
    /*! the actual variable *values* */
    const std::vector<Variable::SP> variables;
    
    /*! our own type description, that tells us which variables (of
      which type, etc) we have */
    std::shared_ptr<SBTObjectType> const type;
  };
  
  template<typename ObjectType>
  struct SBTObject : public SBTObjectBase//RegisteredObject
  {
    typedef std::shared_ptr<SBTObject> SP;

    SBTObject(Context *const context,
              ObjectRegistry &registry,
              std::shared_ptr<ObjectType> type)
      : SBTObjectBase(context,registry,type),
      // : RegisteredObject(context,registry),
        type(type)
      // ,
      //   variables(type->instantiateVariables())
    {
    }
    
    virtual std::string toString() const { return "SBTObject<"+type->toString()+">"; }
    
    /*! our own type description, that tells us which variables (of
      which type, etc) we have */
    std::shared_ptr<ObjectType> const type;
  };

} // ::owl

