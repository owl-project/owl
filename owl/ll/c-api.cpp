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

#ifndef NDEBUG
# define EXCEPTIONS_ARE_FATAL 1
#endif

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
#if EXCEPTIONS_ARE_FATAL
        std::cerr << "Fatal error: " << e.what() << std::endl;
        exit(1);
#else
        lastErrorText = e.what();
        return LLO_UNKNOWN_ERROR;
#endif
      }
    }
    


    extern "C" OWL_LL_INTERFACE
    LLOContext lloContextCreate(const int32_t *deviceIDs,
                                int32_t        numDeviceIDs)
    {
      DeviceGroup *dg = DeviceGroup::create(deviceIDs,numDeviceIDs);
      return (LLOContext)dg;
    }

      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloContextDestroy(LLOContext llo)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           DeviceGroup::destroy(dg);
         });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloLaunch2D(LLOContext llo,
                          int32_t rayGenID,
                          int32_t launchDimX,
                          int32_t launchDimY)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->launch(rayGenID,vec2i(launchDimX,launchDimY));
         });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocModules(LLOContext llo,
                              int numModules)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->allocModules(numModules);
         });
    }


    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocRayGens(LLOContext llo,
                              int32_t    rayGenProgCount)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->allocRayGens(rayGenProgCount);
         });
    }
    
    extern "C" OWL_LL_INTERFACE
    LLOResult lloModuleCreate(LLOContext llo,
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
    extern "C" OWL_LL_INTERFACE
    LLOResult lloBuildModules(LLOContext llo)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->buildModules();
         });
    }

    extern "C" OWL_LL_INTERFACE
    LLOResult lloRayGenCreate(LLOContext  llo,
                              /*! ID of ray gen prog to create */
                              int32_t     programID,
                              /*! ID of module in which to look for that program */
                              int32_t     moduleID,
                              /*! name of the program */
                              const char *programName,
                              /*! size of that program's SBT data */
                              size_t      dataSizeOfRayGen)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->setRayGen(programID,moduleID,programName,dataSizeOfRayGen);
         });
    }
  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloBuildPrograms(LLOContext llo)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->buildPrograms();
         });
    }
  
    extern "C" OWL_LL_INTERFACE
    LLOResult lloCreatePipeline(LLOContext llo)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->createPipeline();
         });
    }
      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloHostPinnedBufferCreate(LLOContext llo,
                                        /*! ID of buffer to create */
                                        int32_t bufferID,
                                        /*! number of elements */
                                        size_t sizeInBytes)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->createHostPinnedBuffer(bufferID,sizeInBytes,1);
         });
    }
      
    extern "C" OWL_LL_INTERFACE
    LLOResult lloAllocBuffers(LLOContext llo,
                              /*! number of buffers valid after this
                               *  function call */
                              int32_t numBuffers)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->allocBuffers(numBuffers);
         });
    }

    /*! builds the SBT's ray gen program entries, using the given
     *  callback to query the app as as to what values to write for a
     *  given ray gen program */
    extern "C" OWL_LL_INTERFACE
    LLOResult lloSbtBuildRayGens(LLOContext llo,
                                 LLOWriteRayGenDataCB writeRayGenDataCB,
                                 const void *callbackData)
    {
      return squashExceptions
        ([&](){
           DeviceGroup *dg = (DeviceGroup *)llo;
           dg->sbtRayGensBuild(owl::ll::WriteRayGenDataCB(writeRayGenDataCB),
                               callbackData);
         });
    }
  
    extern "C" OWL_LL_INTERFACE
    size_t lloGetDeviceCount(LLOContext llo)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->getDeviceCount();
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return size_t(-1);
      }
    }
  
    /*! returns the device-side pointer of the given buffer, on the
     *  given device */
    extern "C" OWL_LL_INTERFACE
    const void *lloBufferGetPointer(LLOContext llo,
                                    int32_t    bufferID,
                                    int32_t    deviceID)
    {
      try {
        DeviceGroup *dg = (DeviceGroup *)llo;
        return dg->bufferGetPointer(bufferID,deviceID);
      } catch (const std::runtime_error &e) {
        lastErrorText = e.what();
        return nullptr;
      }
    }
  

  } // ::owl::ll
} //::owl
