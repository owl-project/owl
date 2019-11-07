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

#include "optix/Context.h"
#include "optix/Module.h"
#include "optix/Program.h"
#include <optix_function_table_definition.h>

namespace optix {

  std::mutex Context::g_mutex;
  
  void Context::log_cb(unsigned int level,
                       const char *tag,
                       const char *message,
                       void * /*cbdata */)
  {
    fprintf( stderr, "#owl: [%2d][%12s]: %s\n", level, tag, message );
  }
  
  /*! first step in the boot-strappin process: globally initialize
    all devices */
  void Context::g_init()
  {
    static bool initialized = false;
    if (initialized) return;
    std::lock_guard<std::mutex> lock(g_mutex);

    // initialize cuda:
    cudaFree(0);
    
    // initialize optix:
    OPTIX_CHECK(optixInit());

    // query number of possible devices:
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    if (numDevices == 0)
      throw Error("Device::g_init","could not find a single CUDA capable device!?");
    
    std::cout << "#owl.context: found " << numDevices << " CUDA-capable devices" << std::endl;
  }

  /*! (try to) create a new device on given ID. may throw an error
    of this device can not be created (eg, if an invalid ID was
    passed, of if there is an error creating an of the optix or
    cuda device contexts */
  Context::PerDevice::SP
  Context::PerDevice::create(int cudaDeviceID,
                             int optixDeviceID,
                             Context *self)
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, cudaDeviceID);
    std::cout << "#owl.device: - device #" << cudaDeviceID
              << ": " << prop.name << std::endl;

    CUcontext cudaContext;
    CUresult res = cuCtxCreate(&cudaContext,0,(CUdevice)cudaDeviceID);
    if (res != CUDA_SUCCESS)
      throw Error("Device::create","error in creating cuda context");
    res = cuCtxSetCurrent(cudaContext);
    if (res != CUDA_SUCCESS)
      throw Error("Device::create","error in activating newly created cuda context");
    
    CUstream stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    OptixDeviceContext optixContext;
    OPTIX_CHECK(optixDeviceContextCreate(cudaContext, 0, &optixContext));

    PerDevice *perDevice
      = new PerDevice(self,cudaDeviceID,optixDeviceID,
                      cudaContext,stream,optixContext);
    assert(perDevice);

    PerDevice::SP sp = PerDevice::SP(perDevice);
    OPTIX_CHECK(optixDeviceContextSetLogCallback
                (optixContext,Context::log_cb,perDevice,4));

    return sp;
  }


  Context::PerDevice::PerDevice(Context *self,
                                int cudaID,
                                int optixID,
                                CUcontext          cudaContext,
                                CUstream           stream,
                                OptixDeviceContext optixContext)
    : self(self),
      cudaID(cudaID),
      optixID(optixID),
      cudaContext(cudaContext),
      stream(stream),
      optixContext(optixContext)
  {
    /* nothing else to do */
  }


    /*! creates a new context with the given device IDs. Invalid
        device IDs get ignored with a warning, but if no device can be
        created at all an error will be thrown. 

        will throw an error if no device(s) could be found for this context

        Should never be called directly, only through Context::create() */
  Context::Context(const std::vector<uint32_t> &deviceIDs)
    : entryPoints(1)    
  {
    Context::g_init();

    if (deviceIDs.empty())
      throw optix::Error("Context::create(deviceIDs)",
                         "context creation called without any devices!?"
                         );
    
    std::cout << "#owl.context: creating new owl context..." << std::endl;
    perDevice.resize(deviceIDs.size());
    for (int i=0;i<deviceIDs.size();i++)
      perDevice[i] = PerDevice::create(i,deviceIDs[i],this);

    // -------------------------------------------------------
    // initialize shared state data to default values 
    // -------------------------------------------------------
    initializePipelineDefaults();
  }
    
  /*! should only once be called by the constructor, to initialize all
    compile/link options to defaults */
  void Context::initializePipelineDefaults()
  {
    // ------------------------------------------------------------------
    // module compile options
    // ------------------------------------------------------------------
    moduleCompileOptions.maxRegisterCount  = 100;
    moduleCompileOptions.optLevel          = OPTIX_COMPILE_OPTIMIZATION_LEVEL_0;
    moduleCompileOptions.debugLevel        = OPTIX_COMPILE_DEBUG_LEVEL_LINEINFO;//FULL;

    
    // ------------------------------------------------------------------
    // pipeline compile options
    // ------------------------------------------------------------------
    pipelineCompileOptions.traversableGraphFlags
      // = instances.empty()
      // ? OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS
      // : OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
      = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    
    pipelineCompileOptions.usesMotionBlur     = false;
    pipelineCompileOptions.numPayloadValues   = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags     = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "optixLaunchParams";
  }
  
  /*! creates a new context with the given device IDs. Invalid
    device IDs get ignored with a warning, but if no device can be
    created at all an error will be thrown */
  Context::SP Context::create(const std::vector<uint32_t> &deviceIDs)
  {
    return std::make_shared<Context>(deviceIDs);
  }

  /*! creates a new context with the given device IDs. Invalid
    device IDs get ignored with a warning, but if no device can be
    created at all an error will be thrown */
  Context::SP Context::create(GPUSelectionMethod whichGPUs)
  {
    switch(whichGPUs) {
    case Context::GPU_SELECT_FIRST:
      return create((std::vector<uint32_t>){0});
    default:
      throw optix::Error("Context::create(GPUSelectionMethod)",
                         "specified GPU Selection method not implemented "
                         "(method="+std::to_string((int)whichGPUs)+")"
                         );
    }
  }


  /*! create a new module object from given ptx string */
  ModuleSP  Context::createModuleFromString(const std::string &ptxCode)
  {
    // TODO: add some check here that checks if a module with that
    // string was already created, and if so, emit a warning if in
    // profile mode
    return Module::create(this,ptxCode);
  }
    

  /*! set raygen program name and module for given entry point */
  void Context::setEntryPoint(size_t entryPointID,
                              ModuleSP module,
                              const std::string &programName)
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (entryPointID >= entryPoints.size())
      throw Error("Context::setEntryPoint",
                  "invalid entry point ID"
                  " - did you call Context::setNumEntryPoints()?");

    entryPoints[entryPointID]
      = std::make_shared<Program>(module,programName);
  }
  
} // ::optix
