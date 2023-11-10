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

#include "pyOWL/Module.h"
#include "pyOWL/Context.h"
#include <streambuf>
#include <sstream>
#include <fstream>
#include <string.h>

namespace pyOWL {

  template<typename Lambda>
  bool checkTag(const std::string &line,
                const char *tag,
                const Lambda &lambda)
  {
    const char *where = strstr(line.c_str(),tag);
    if (!where) return false;
    lambda(where+strlen(tag));
    return true;
  }
  
  Module::Module(std::shared_ptr<Context> ctx,
                 const std::string &ptx)
    : context(ctx)
  {
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "#pyOWL: creating module..."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    
    std::stringstream purged;
    
    std::istringstream in(ptx);
    std::string line;
    while (std::getline(in,line)) {
      if (checkTag(line,"__owl_varDeclOffset__",
                   [this](const char *decl)
                   { addVarDeclOffset(decl); }))
        continue;
      
      if (checkTag(line,"__owl_varDeclType__",
               [this](const char *decl)
               { addVarDeclType(decl); }))
        continue;
      
      if (checkTag(line,"__owl_typeDecl__",
                   [this](const char *decl)
                   { addTypeDecl(decl); }))
        continue;
      
      purged << line << std::endl;
    }
    ptxCode = purged.str();
    std::cout << OWL_TERMINAL_BLUE
              << "#pyOWL: done parsing module PTX file, found "
              << typeDecls.size() << " type declarations"
              << OWL_TERMINAL_DEFAULT
              << std::endl;

    handle = owlModuleCreate(ctx->handle,ptxCode.c_str());
    std::cout << OWL_TERMINAL_GREEN
              << "#pyOWL: module created."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
  }
  
  size_t Module::getTypeSize(const std::string &typeName) const
  {
    auto it = typeDecls.find(typeName);
    if (it == typeDecls.end())
      throw std::runtime_error("could not find type declarations for type '"+typeName+"'");
    return it->second.size;
  }
  
  std::vector<OWLVarDecl> Module::getTypeVars(const std::string &typeName) const
  {
    auto it = typeDecls.find(typeName);
    if (it == typeDecls.end())
      throw std::runtime_error("could not find pyOWL type variables for type '"
                               +typeName+"'");

    const TypeDecl &type = it->second;
    std::vector<OWLVarDecl> varDecls;
    for (const auto &var : type.varDecls) {
      OWLVarDecl decl
        = {
           var.first.c_str(),
           var.second.type,
           (uint32_t)var.second.offset
      };
      // std::cout << " var " << decl.name << " type " << decl.type << " ofs " << decl.offset << std::endl;
      varDecls.push_back(decl);
    }
    return varDecls;
  }

  Module::~Module()
  {
    if (context->alive())
      owlModuleRelease(handle);
    handle = 0;
  }

  /*! internal helper function - using a line from the ptx file
    (that was generated with the PYOWL_DECLARE_TYPE macro -
    extract the size of the given type,a nd record i tin the
    typeDecl map */
  void Module::addTypeDecl(const char *begin)
  {
    const char *end = begin;
    while (*end && *end != ' ') ++end;
    const std::string typeName(begin,end);

    int typeSize = 0;
    sscanf(end," = %i;",&typeSize);
    std::cout << OWL_TERMINAL_LIGHT_BLUE
              << "#pyOWL: detected type decl of type " << typeName << ", sizeof(" << typeName
              << ") = " << typeSize
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    typeDecls[typeName].size = typeSize;
  }

  void parseVarDecl(const char *line,
                    std::string &className,
                    std::string &varName,
                    uint32_t    &value)
  {
    // -------------------------------------------------------
    const char *classNameBegin = line;
    const char *classNameEnd   = strstr(classNameBegin,"____");
    if (!classNameEnd)
      throw std::runtime_error("could not parse variable declaration (2)");
    className = std::string(classNameBegin,classNameEnd-classNameBegin);

    // -------------------------------------------------------
    const char *varNameBegin  = classNameEnd+4;
    const char *varNameEnd    = varNameBegin;
    while (*varNameEnd && *varNameEnd != ' ' && *varNameEnd != ';')
      ++varNameEnd;
    varName = std::string (varNameBegin,varNameEnd-varNameBegin);
    
    // -------------------------------------------------------
    if (*varNameEnd == ';')
      value = 0;
    else {
      const char *valueBegin = varNameEnd + strlen(" = ");
      const char *valueEnd = strstr(valueBegin,";");
      value = std::stol(std::string(valueBegin,valueEnd));
    }
  }
                    
  /*! internal helper function - using a line from the ptx file
    (that was generated with the PYOWL_DECLARE_VAR macro - extract
    the size of the given var,a nd record i tin the varDecl map */
  void Module::addVarDeclOffset(const char *decl)
  {
    std::string type, var;
    uint32_t val;
    parseVarDecl(decl,type,var,val);
    
    typeDecls[type].varDecls[var].offset = val;
  }
  
  /*! internal helper function - using a line from the ptx file
    (that was generated with the PYOWL_DECLARE_VAR macro - extract
    the size of the given var,a nd record i tin the varDecl map */
  void Module::addVarDeclType(const char *decl)
  {
    std::string type, var;
    uint32_t val;
    parseVarDecl(decl,type,var,val);
    
    typeDecls[type].varDecls[var].type = (OWLDataType)val;
  }
  
  /*! create a new PTX Module from ptx file indicated by given fileName */
  std::shared_ptr<Module> Module::fromFile(std::shared_ptr<Context> ctx,
                                           const std::string &fileName)
  {
    std::ifstream in(fileName);
    if (!in.good())
      throw std::runtime_error("could not open PTX file '"+fileName+"'");
    std::string ptx((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
    return std::make_shared<Module>(ctx,ptx);
  }

  /*! creates a new PTX module directly from a string containting
    the PTX code */
  std::shared_ptr<Module> Module::fromString(std::shared_ptr<Context> ctx,
                                             const std::string &ptx)
  {
    return std::make_shared<Module>(ctx,ptx);
  }
  
}
