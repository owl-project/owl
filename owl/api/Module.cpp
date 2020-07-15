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

#include "Module.h"
#include "Context.h"

#define LOG(message)                            \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_LIGHT_BLUE                \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl

#define LOG_OK(message)                         \
  if (ll::Context::logging())                   \
    std::cout                                   \
      << OWL_TERMINAL_BLUE                      \
      << "#owl.ng: "                            \
      << message                                \
      << OWL_TERMINAL_DEFAULT << std::endl


namespace owl {

  inline bool ptxContainsInvalidOptixInternalCall(const std::string &line)
  {
    static const char *optix_internal_symbols[] = {
                                                   " _optix_",
                                                   nullptr
    };
    for (const char **testSym = optix_internal_symbols; *testSym; ++testSym) {
      if (line.find(*testSym) != line.npos)
        return true;
    }
    return false;
  }
    
  std::string getNextLine(const char *&s)
  {
    std::stringstream line;
    while (*s) {
      char c = *s++;
      line << c;
      if (c == '\n') break;
    }
    return line.str();
  }
    
  std::string killAllInternalOptixSymbolsFromPtxString(const char *orignalPtxCode)
  {
    std::stringstream fixed;

    for (const char *s = orignalPtxCode; *s; ) {
      std::string line = getNextLine(s);
      if (ptxContainsInvalidOptixInternalCall(line))
        fixed << "//dropped: " << line;
      else
        fixed << line;
    }
    return fixed.str();
  }


  Module::Module(Context *const context,
                 const std::string &ptxCode)
    : RegisteredObject(context,context->modules),
      ptxCode(ptxCode)
  {
    // lloModuleCreate(context->llo,this->ID,
    // context->llo->moduleCreate(this->ID,
    //                            // warning: this 'this' here is importat, since
    //                            // *we* manage the lifetime of this string, and
    //                            // the one on the constructor list will go out of
    //                            // scope after this function
    //                            this->ptxCode.c_str());
  }

  void Module::DeviceData::build(Module *parent, Device *device)
  {
    assert(module == 0);
    const int oldActive = device->pushActive();
    
    LOG("building module #" + parent->toString());
    
    char log[2048];
    size_t sizeof_log = sizeof( log );

    OPTIX_CHECK_LOG(optixModuleCreateFromPTX(device->context->optixContext,
                                             &device->context->moduleCompileOptions,
                                             &device->context->pipelineCompileOptions,
                                             parent->ptxCode.c_str(),
                                             strlen(parent->ptxCode.c_str()),
                                             log,      // Log string
                                             &sizeof_log,// Log string sizse
                                             &module
                                             ));
    assert(module != nullptr);

    // ------------------------------------------------------------------
    // Now, build separate cuda-only module that does not contain
    // any optix-internal symbols. Note this does not actually
    // *remove* any potentially existing anyhit/closesthit/etc.
    // programs in this module - it just removed all optix-related
    // calls from this module, but leaves the remaining (now
    // dysfunctional) anyhit/closesthit/etc. programs still in that
    // PTX code. It would obviously be cleaner to completely
    // remove those programs, but that would require significantly
    // more advanced parsing of the PTX string, so right now we'll
    // just leave them in (and as it's in a module that never gets
    // used by optix, this should actually be OK).
    // ------------------------------------------------------------------
    const char *ptxCode = parent->ptxCode.c_str();
    LOG("generating second, 'non-optix' version of that module, too");
    CUresult rc = (CUresult)0;
    const std::string fixedPtxCode
      = killAllInternalOptixSymbolsFromPtxString(parent->ptxCode.c_str());
    strcpy(log,"(no log yet)");
    CUjit_option options[] = {
                              CU_JIT_TARGET_FROM_CUCONTEXT,
                              CU_JIT_ERROR_LOG_BUFFER,
                              CU_JIT_ERROR_LOG_BUFFER_SIZE_BYTES,
    };
    void *optionValues[] = {
                            (void*)0,
                            (void*)log,
                            (void*)sizeof(log)
    };
    rc = cuModuleLoadDataEx(&boundsModule, (void *)fixedPtxCode.c_str(),
                            3, options, optionValues);
    if (rc != CUDA_SUCCESS) {
      const char *errName = 0;
      cuGetErrorName(rc,&errName);
      PRINT(errName);
      PRINT(log);
      exit(0);
    }
    LOG_OK("created module #" << parent->ID << " (both optix and cuda)");
    device->popActive(oldActive);
  }
  
  void Module::DeviceData::destroy(Device *device)
  {
    const int oldActive = device->pushActive();
    
    assert(module);
    optixModuleDestroy(module);
    module = 0;

    device->popActive(oldActive);
  }

} // ::owl
