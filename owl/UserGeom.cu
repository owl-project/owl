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

#define FREE_EARLY 1

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


                                          
  /*! construct a new device-data for this type */
  UserGeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : GeomType::DeviceData(device)
  {}

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
  void UserGeom::setNumMotionKeys(uint32_t _numKeys)
  {
    numKeys = _numKeys;
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

  /*! run the bounding box program for all primitives within this geometry */
  void UserGeom::executeBoundsProgOnPrimitives(const DeviceContext::SP &device)
  {
    SetActiveGPU activeGPU(device);
    // if geom does't contain any prims we would run into issue
    // launching zero-sized bounds prog kernel below, so let's just
    // exit here.
    if (primCount == 0) return;
    DeviceData &dd = getDD(device);
      
    std::vector<uint8_t> userGeomData(geomType->varStructSize);
    
    if (dd.tempMem.sizeInBytes < geomType->varStructSize) 
    {
      if (!dd.tempMem.empty()) dd.tempMem.free();
      dd.tempMem.alloc(geomType->varStructSize);
    }
    
    dd.internalBufferForBoundsProgram.resize(numKeys);
    for (uint32_t i = 0; i < numKeys; ++i) {
      if (dd.internalBufferForBoundsProgram[i].sizeInBytes < primCount*sizeof(box3f)) 
      {
        if (!dd.internalBufferForBoundsProgram[i].empty()) dd.internalBufferForBoundsProgram[i].free();
        dd.internalBufferForBoundsProgram[i].alloc(primCount*sizeof(box3f));
      }
    }
    
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

    dd.tempMem.upload(userGeomData);
    
    void  *d_geomData = dd.tempMem.get();
    for (int k = 0; k < numKeys; ++k) {
      void* d_boundsArray = dd.internalBufferForBoundsProgram[k].get();

      /* arguments for the kernel we are to call */
      void  *args[] = {
                      &d_geomData,
                      &d_boundsArray,
                      (void *)&primCount,
                      (void *)&k
      };
      
      CUstream stream = device->stream;
      UserGeomType::DeviceData &typeDD = getTypeDD(device);
      if (!typeDD.boundsFuncKernel)
        OWL_RAISE("bounds kernel set, but not yet compiled - "
                                "did you forget to call BuildPrograms() before"
                                " (User)GroupAccelBuild()!?");
          
      CUresult rc
        = cuLaunchKernel(typeDD.boundsFuncKernel,
                        gridDims.x,gridDims.y,gridDims.z,
                        blockDims.x,blockDims.y,blockDims.z,
                        0, stream, args, 0);
      if (rc) {
        const char *errName = 0;
        cuGetErrorName(rc,&errName);
        OWL_RAISE("unknown CUDA error in calling bounds function kernel: "
                                +std::string(errName));
      }
    }
    
    #ifdef FREE_EARLY
    // only free temporary memory if necessary. For realtime builds, instead recycle if possible
    dd.tempMem.free(); 
    cudaDeviceSynchronize();
    #endif
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
      
      assert(moduleDD.boundsModule);

      const std::string annotatedProgName
        = std::string("__boundsFuncKernel__")
        + boundsProg.progName;
    
      CUresult rc = cuModuleGetFunction(&typeDD.boundsFuncKernel,
                                        moduleDD.boundsModule,
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
        cuGetErrorName(rc,&errName);
        OWL_RAISE("unknown CUDA error when building bounds program kernel"
                  +std::string(errName));
      }
    }
  }

} // ::owl
