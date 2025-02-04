// ======================================================================== //
// Copyright 2019-2021 Ingo Wald                                            //
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

#include "UserGeom.h"
#include "Context.h"
#include "CUDADriver.h"

namespace owl {

#define LOG(message)                                    \
  if (Context::logging())                               \
    std::cout << "#owl(" << device->ID << "): "         \
              << message                                \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl(" << device->ID << "): "                 \
              << message << OWL_TERMINAL_DEFAULT << std::endl

  __device__ static float atomicMax(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fmaxf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __device__ static float atomicMin(float* address, float val)
  {
    int* address_as_i = (int*) address;
    int old = *address_as_i, assumed;
    do {
      assumed = old;
      old = ::atomicCAS(address_as_i, assumed,
                        __float_as_int(::fminf(val, __int_as_float(assumed))));
    } while (assumed != old);
    return __int_as_float(old);
  }
  
  __global__ void computeBoundsOfPrimBounds(box3f *d_bounds,
                                            const box3f *d_primBounds,
                                            size_t count)
  {
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid >= count) return;

    box3f box = d_primBounds[tid];
    if (!box.empty()) {
      atomicMin(&d_bounds->lower.x,box.lower.x);
      atomicMin(&d_bounds->lower.y,box.lower.y);
      atomicMin(&d_bounds->lower.z,box.lower.z);
      atomicMax(&d_bounds->upper.x,box.upper.x);
      atomicMax(&d_bounds->upper.y,box.upper.y);
      atomicMax(&d_bounds->upper.z,box.upper.z);
    }
  }
                                          
  /*! construct a new device-data for this type */
  UserGeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : GeomType::DeviceData(device)
  {}

  
  UserGeom::DeviceData::~DeviceData()
  {
  }
  
  UserGeomType::DeviceData::~DeviceData()
  {
  }
  
  
  std::shared_ptr<Geom> UserGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<UserGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }
  
  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  UserGeom::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

  /*! create this object's device-specific data for the device */
  RegisteredObject::DeviceData::SP
  UserGeomType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

  /*! pretty-printer, for printf-debugging */
  std::string UserGeomType::toString() const
  { return "UserGeomType"; }

  /*! pretty-printer, for printf-debugging */
  std::string UserGeom::toString() const
  { return "UserGeom"; }
  
  UserGeomType::UserGeomType(Context *const context,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls),
      intersectProg(context->numRayTypes)
  {
    /*! nothing special - all inherited */
  }

  UserGeomType::~UserGeomType()
  {
  }
  
  /*! constructor */
  UserGeom::UserGeom(Context *const context,
                     GeomType::SP geometryType)
    : Geom(context,geometryType)
  {}


  /*! set number of primitives that this geom will contain */
  void UserGeom::setPrimCount(size_t count)
  {
    primCount = count;
  }

  void UserGeom::setBoundsBuffer(Buffer::SP buffer)
  {
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.internalBufferForBoundsProgram.d_pointer
        = (CUdeviceptr)buffer->getPointer(device);
      dd.internalBufferForBoundsProgram.sizeInBytes
        = buffer->sizeInBytes();
      dd.internalBufferForBoundsProgram.externallyManaged
        = true;
      dd.useExternalBoundsBuffer = true;
    }
  }

  void UserGeom::setMotionBoundsBuffers(Buffer::SP buffer1, Buffer::SP buffer2)
  {
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.useExternalBoundsBuffer = true;
      dd.internalBufferForBoundsProgramKey1.d_pointer = (CUdeviceptr)buffer1->getPointer(device);
      dd.internalBufferForBoundsProgramKey1.sizeInBytes = buffer1->sizeInBytes();
      dd.internalBufferForBoundsProgramKey1.externallyManaged = true;

      dd.internalBufferForBoundsProgramKey2.d_pointer = (CUdeviceptr)buffer2->getPointer(device);
      dd.internalBufferForBoundsProgramKey2.sizeInBytes = buffer2->sizeInBytes();
      dd.internalBufferForBoundsProgramKey2.externallyManaged = true;
    }
  }

  /*! set intersection program to run for this type and given ray type */
  void UserGeomType::setIntersectProg(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType >= 0 && rayType < intersectProg.size());

    intersectProg[rayType].progName = "__intersection__"+progName;
    intersectProg[rayType].module   = module;
  }

  /*! set bounding box program to run for this type */
  void UserGeomType::setBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->boundsProg.progName = progName;
    this->boundsProg.module   = module;
  }

  /*! set bounding box program to run for this type */
  void UserGeomType::setMotionBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->motionBoundsProg.progName = progName;
    this->motionBoundsProg.module   = module;
  }

  /*! run the bounding box program for all primitives within this geometry */
  void UserGeom::executeBoundsProgOnPrimitives(const DeviceContext::SP &device)
  {
    SetActiveGPU activeGPU(device);
    // if geom does't contain any prims we would run into issue
    // launching zero-sized bounds prog kernel below, so let's just
    // exit here.
    if (primCount == 0) return;
      
    std::vector<uint8_t> userGeomData(geomType->varStructSize);
    
    DeviceData &dd = getDD(device);
    
    if (!dd.useExternalBoundsBuffer)
      dd.internalBufferForBoundsProgram.alloc(primCount*sizeof(box3f));
    else if (dd.internalBufferForBoundsProgram.size() < primCount*sizeof(box3f))
      OWL_RAISE("external bounds buffer size does not match primCount");

    DeviceMemory &tempMem = dd.tempMem;
    tempMem.alloc(geomType->varStructSize);

    writeVariables(userGeomData.data(),device);
        
    // size of each thread block during bounds function call
    vec3i blockDims(32,32,1);
    uint32_t threadsPerBlock = blockDims.x*blockDims.y*blockDims.z;
        
    uint32_t numBlocks = owl::common::divRoundUp((uint32_t)primCount,threadsPerBlock);
    uint32_t numBlocks_x
      = 1+uint32_t(powf((float)numBlocks,1.f/3.f));
    uint32_t numBlocks_y
      = 1+uint32_t(sqrtf((float)(numBlocks/numBlocks_x)));
    uint32_t numBlocks_z
      = owl::common::divRoundUp(numBlocks,numBlocks_x*numBlocks_y);
        
    vec3i gridDims(numBlocks_x,numBlocks_y,numBlocks_z);

    tempMem.upload(userGeomData);
    
    void  *d_geomData = tempMem.get();
    vec3f *d_boundsArray = (vec3f*)dd.internalBufferForBoundsProgram.get();
    /* arguments for the kernel we are to call */
    void  *args[] = {
                     &d_geomData,
                     &d_boundsArray,
                     (void *)&primCount
    };

    CUstream stream = device->stream;
    UserGeomType::DeviceData &typeDD = getTypeDD(device);
    if (!typeDD.boundsFuncKernel)
      OWL_RAISE("bounds kernel set, but not yet compiled - "
                "did you forget to call BuildPrograms() before"
                " (User)GroupAccelBuild()!?");
        
    CUresult rc
      = _cuLaunchKernel(typeDD.boundsFuncKernel,
                       gridDims.x,gridDims.y,gridDims.z,
                       blockDims.x,blockDims.y,blockDims.z,
                       0, stream, args, 0);
    if (rc) {
      const char *errName = 0;
      _cuGetErrorName(rc,&errName);
      OWL_RAISE("unknown CUDA error in calling bounds function kernel: "
                +std::string(errName));
    }
    
    cudaDeviceSynchronize();
  }

  /*! run the motion bounding box program for all primitives within this geometry */
  void UserGeom::executeMotionBoundsProgOnPrimitives(const DeviceContext::SP &device)
  {
    SetActiveGPU activeGPU(device);
    // if geom does't contain any prims we would run into issue
    // launching zero-sized bounds prog kernel below, so let's just
    // exit here.
    if (primCount == 0) return;
      
    std::vector<uint8_t> userGeomData(geomType->varStructSize);
       
    DeviceData &dd = getDD(device);
    if (!dd.useExternalBoundsBuffer) {
      dd.internalBufferForBoundsProgramKey1.alloc(primCount*sizeof(box3f));
      dd.internalBufferForBoundsProgramKey2.alloc(primCount*sizeof(box3f));
    }
    else {
      if (dd.internalBufferForBoundsProgramKey1.size() < primCount*sizeof(box3f))
        OWL_RAISE("external bounds buffer for key 0 size does not match primCount");
      if (dd.internalBufferForBoundsProgramKey2.size() < primCount*sizeof(box3f))
        OWL_RAISE("external bounds buffer for key 1 size does not match primCount");
    } 

    DeviceMemory &tempMem = dd.tempMem;
    tempMem.alloc(geomType->varStructSize);

    writeVariables(userGeomData.data(),device);
        
    // size of each thread block during bounds function call
    vec3i blockDims(32,32,1);
    uint32_t threadsPerBlock = blockDims.x*blockDims.y*blockDims.z;
        
    uint32_t numBlocks = owl::common::divRoundUp((uint32_t)primCount,threadsPerBlock);
    uint32_t numBlocks_x
      = 1+uint32_t(powf((float)numBlocks,1.f/3.f));
    uint32_t numBlocks_y
      = 1+uint32_t(sqrtf((float)(numBlocks/numBlocks_x)));
    uint32_t numBlocks_z
      = owl::common::divRoundUp(numBlocks,numBlocks_x*numBlocks_y);
        
    vec3i gridDims(numBlocks_x,numBlocks_y,numBlocks_z);

    tempMem.upload(userGeomData);

    void  *d_geomData = tempMem.get();
    vec3f *d_boundsArrayKey1 = (vec3f*)dd.internalBufferForBoundsProgramKey1.get();
    vec3f *d_boundsArrayKey2 = (vec3f*)dd.internalBufferForBoundsProgramKey2.get();
    /* arguments for the kernel we are to call */
    void  *args[] = {
                     &d_geomData,
                     &d_boundsArrayKey1,
                     &d_boundsArrayKey2,
                     (void *)&primCount
    };
    
    CUstream stream = device->stream;
    UserGeomType::DeviceData &typeDD = getTypeDD(device);
    if (!typeDD.motionBoundsFuncKernel)
      OWL_RAISE("bounds kernel set, but not yet compiled - "
                "did you forget to call BuildPrograms() before"
                " (User)GroupAccelBuild()!?");

    CUresult rc
      = _cuLaunchKernel(typeDD.motionBoundsFuncKernel,
                        gridDims.x,gridDims.y,gridDims.z,
                        blockDims.x,blockDims.y,blockDims.z,
                        0, stream, args, 0);
    if (rc) {
      const char *errName = 0;
      _cuGetErrorName(rc,&errName);
      OWL_RAISE("unknown CUDA error in calling motion bounds function kernel: "
                +std::string(errName));
    }
    
    cudaDeviceSynchronize();
  }

  /*! fill in an OptixProgramGroup descriptor with the module and
    program names for this type */
  void UserGeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                            GeomType *_parent,
                                            int rt)
  {
    GeomType::DeviceData::fillPGDesc(pgDesc,_parent,rt);
    UserGeomType *parent = (UserGeomType*)_parent;
    
    // ----------- intserect -----------
    if (rt < (int)parent->intersectProg.size()) {
      const ProgramDesc &pd = parent->intersectProg[rt];
      if (pd.module) {
        pgDesc.hitgroup.moduleIS = pd.module->getDD(device).module;
        pgDesc.hitgroup.entryFunctionNameIS = pd.progName.c_str();
      }
    }
  }
  
  /*! build the CUDA bounds program kernel (if bounds prog is set) */
  void UserGeomType::buildBoundsProg()
  {
    if (!boundsProg.module) return;
    
    Module::SP module = boundsProg.module;
    assert(module);

    for (auto device : context->getDevices()) {
      LOG("building bounds function ....");
      SetActiveGPU forLifeTime(device);
      auto &typeDD = getDD(device);
      auto &moduleDD = module->getDD(device);
      
      assert(moduleDD.computeModule);

      const std::string annotatedProgName
        = std::string("__boundsFuncKernel__")
        + boundsProg.progName;
    
      CUresult rc = _cuModuleGetFunction(&typeDD.boundsFuncKernel,
                                        moduleDD.computeModule,
                                        annotatedProgName.c_str());
      
      switch(rc) {
      case CUDA_SUCCESS:
        /* all OK, nothing to do */
        LOG_OK("found bounds function " << annotatedProgName << " ... perfect!");
        break;
      case CUDA_ERROR_NOT_FOUND:
        OWL_RAISE("in "+std::string(__PRETTY_FUNCTION__)
                  +": could not find OPTIX_BOUNDS_PROGRAM("
                  +boundsProg.progName+")");
      default:
        const char *errName = 0;
        _cuGetErrorName(rc,&errName);
        OWL_RAISE("unknown CUDA error when building bounds program kernel"
                  +std::string(errName));
      }
    }
  }

  /*! build the CUDA bounds program kernel (if bounds prog is set) */
  void UserGeomType::buildMotionBoundsProg()
  {
    if (!motionBoundsProg.module) return;
    
    Module::SP module = motionBoundsProg.module;
    assert(module);

    for (auto device : context->getDevices()) {
      LOG("building motion bounds function ....");
      SetActiveGPU forLifeTime(device);
      auto &typeDD = getDD(device);
      auto &moduleDD = module->getDD(device);
      
      assert(moduleDD.computeModule);

      const std::string annotatedProgName
        = std::string("__motionBoundsFuncKernel__")
        + motionBoundsProg.progName;
    
      CUresult rc = _cuModuleGetFunction(&typeDD.motionBoundsFuncKernel,
                                        moduleDD.computeModule,
                                        annotatedProgName.c_str());
      
      switch(rc) {
      case CUDA_SUCCESS:
        /* all OK, nothing to do */
        LOG_OK("found motion bounds function " << annotatedProgName << " ... perfect!");
        break;
      case CUDA_ERROR_NOT_FOUND:
        OWL_RAISE("in "+std::string(__PRETTY_FUNCTION__)
                  +": could not find OPTIX_MOTION_BOUNDS_PROGRAM("
                  +motionBoundsProg.progName+")");
      default:
        const char *errName = 0;
        _cuGetErrorName(rc,&errName);
        OWL_RAISE("unknown CUDA error when building motion bounds program kernel"
                  +std::string(errName));
      }
    }
  }

} // ::owl
