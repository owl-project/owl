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

/*! \file include/owl/ll.h Implements the dynamically linkable "C-API"
 *  for the low-level owl::ll abstraction layer */

// public API:
#include "../include/owl/ll.h"
// internal C++ classes that implement this API
#include "owl/ll/DeviceGroup.h"

// #undef OWL_LL_INTERFACE
// #define OWL_LL_INTERFACE extern "C"

namespace owl {
  namespace ll {

    std::string lastErrorText = "";

    template<typename Lambda>
    inline LLOResult squashExceptions(const Lambda &fun)
    {
      try {
        fun();
        return LLO_SUCCESS;
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return LLO_UNKNOWN_ERROR;
      }
    }
    


    extern "C" {
      OWL_LL_INTERFACE LLOContext lloCreate(const int32_t *deviceIDs,
                                            int32_t numDeviceIDs)
    {
      DeviceGroup *dg = DeviceGroup::create(deviceIDs,numDeviceIDs);
      return (LLOContext)dg;
    }

    OWL_LL_INTERFACE LLOResult lloAllocModules(LLOContext llo,
                                               int numModules)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->allocModules(numModules);
         });
    }
  
    OWL_LL_INTERFACE LLOResult lloModuleCreate(LLOContext llo,
                                               int32_t moduleID,
                                               const char *ptxCode)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->setModule(moduleID,ptxCode);
         });
    }

    /*! (re-)builds the modules that have been set via
     *  lloModuleCreate */
    OWL_LL_INTERFACE LLOResult lloBuildModules(LLOContext llo)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->buildModules();
         });
    }
  
    } // extern C
    
  } // ::owl::ll
} //::owl
