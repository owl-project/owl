// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include "pyOWL/common.h"

namespace pyOWL {

  struct Context;
  
  struct Module : public std::enable_shared_from_this<Module>
  {
    typedef std::shared_ptr<Module> SP;

    Module(std::shared_ptr<Context> ctx,
           const std::string &ptx);
    ~Module();

    /*! create a new PTX Module from ptx file indicated by given fileName */
    static std::shared_ptr<Module> fromFile(std::shared_ptr<Context> ctx,
                                            const std::string &fileName);

    /*! creates a new PTX module directly from a string containting
        the PTX code */
    static std::shared_ptr<Module> fromString(std::shared_ptr<Context> ctx,
                                              const std::string &ptx);

    size_t getTypeSize(const std::string &typeName) const;
    std::vector<OWLVarDecl> getTypeVars(const std::string &typeName) const;
    OWLModule getHandle() const { return handle; }
    
  private:
    /*! internal helper function - using a line from the ptx file
        (that was generated with the PYOWL_DECLARE_TYPE macro -
        extract the size of the given type,a nd record i tin the
        typeDecl map */
    void addTypeDecl(const char *decl);
    
    /*! internal helper function - using a line from the ptx file
        (that was generated with the PYOWL_DECLARE_VAR macro - extract
        the size of the given var,a nd record i tin the varDecl map */
    void addVarDeclOffset(const char *decl);
    void addVarDeclType(const char *decl);

    /*! resulting ptx code after the variable declations have been
        stripped out */
    std::string ptxCode;
    
    /*! variable declarations that were contained in the ptx string
        (created with the PYOWL_DECLARE_VARIABLE macro in the .cu
        device code). First map is over the type name, second one
        about the variable name; inner value si the byte offset in the
        type struct */
    std::map<std::string,std::map<std::string,std::pair<OWLDataType,uint64_t>>> varDecls;
    
    /*! type declarations that were contained in the ptx string
        (created with the PYOWL_DECLARE_TYPE macro in the .cu device
        code) */
    struct TypeDecl {
      struct VarDecl { OWLDataType type; size_t offset; };
      
      std::map<std::string,VarDecl> varDecls;
      size_t size = 0;
    };
    std::map<std::string,TypeDecl> typeDecls;

    /*! the owl module handle */
    OWLModule handle = 0;
    std::shared_ptr<Context> context;
  };
  
}
