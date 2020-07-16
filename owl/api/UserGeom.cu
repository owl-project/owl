// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

#include "api/UserGeom.h"
#include "api/Context.h"
#include "ll/DeviceMemory.h"
#include "ll/Device.h"
#include "ll/DeviceGroup.h"
#include "Context.h"

namespace owl {

#define LOG(message)                                            \
  if (Context::logging())                                   \
    std::cout << "#owl.ll(" << device->ID << "): "    \
              << message                                        \
              << std::endl

#define LOG_OK(message)                                         \
  if (Context::logging())                                       \
    std::cout << OWL_TERMINAL_GREEN                             \
              << "#owl.ll(" << device->ID << "): "    \
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
                                          
  /*! call a cuda kernel that computes the bounds of the vertex buffers */
  void UserGeom::computeBounds(box3f bounds[2])
  {
    int numThreads = 1024;
    int numBlocks = (primCount + numThreads - 1) / numThreads;

    ll::Device *device = context->llo->devices[0];
    int oldActive = device->pushActive();

    ll::DeviceMemory d_bounds;
    d_bounds.alloc(sizeof(box3f));
    bounds[0] = bounds[1] = box3f();
    d_bounds.upload(bounds);

    DeviceData &dd = getDD(device);
    // ll::UserGeom *ug =
    //   (ll::UserGeom *)context->llo->devices[0]->checkGetGeom(this->ID);
    
    computeBoundsOfPrimBounds<<<numBlocks,numThreads>>>
      (((box3f*)d_bounds.get())+0,
       (box3f *)dd.internalBufferForBoundsProgram.get(),
       primCount);
    CUDA_SYNC_CHECK();
    d_bounds.download(&bounds[0]);
    d_bounds.free();
    CUDA_SYNC_CHECK();
    bounds[1] = bounds[0];

    device->popActive(oldActive);
  }

  UserGeomType::UserGeomType(Context *const context,
                             size_t varStructSize,
                             const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls),
      intersectProg(context->numRayTypes)
  {
    /*! nothing special - all inherited */
  }

  UserGeom::UserGeom(Context *const context,
                     GeomType::SP geometryType)
    : Geom(context,geometryType)
  {
    // int numPrims = 0;
    // context->llo->userGeomCreate(this->ID,geometryType->ID,numPrims);
  }

  void UserGeom::setPrimCount(size_t count)
  {
    primCount = count;
  }

  void UserGeomType::setIntersectProg(int rayType,
                                      Module::SP module,
                                      const std::string &progName)
  {
    assert(rayType < intersectProg.size());

    intersectProg[rayType].progName = "__intersection__"+progName;
    intersectProg[rayType].module   = module;
    // context->llo->setGeomTypeIntersect(this->ID,
    //                      rayType,module->ID,
    //                      // warning: this 'this' here is importat, since
    //                      // *we* manage the lifetime of this string, and
    //                      // the one on the constructor list will go out of
    //                       // scope after this function
    //                      intersectProg[rayType].progName.c_str());
  }

  void UserGeomType::setBoundsProg(Module::SP module,
                                   const std::string &progName)
  {
    this->boundsProg.progName = progName;
    this->boundsProg.module   = module;
    // context->llo->setGeomTypeBoundsProgDevice(this->ID,
    //                             module->ID,
    //                             // warning: this 'this' here is importat, since
    //                             // *we* manage the lifetime of this string, and
    //                             // the one on the constructor list will go out of
    //                             // scope after this function
    //                             this->boundsProg.progName.c_str(),
    //                             varStructSize
    //                             );
  }


  /*! run the bounding box program for all primitives within this geometry */
  void UserGeom::executeBoundsProgOnPrimitives(ll::Device *device)
  {
    int oldActive = device->pushActive();
      
    std::vector<uint8_t> userGeomData(geomType->varStructSize);
    DeviceMemory tempMem;
    tempMem.alloc(geomType->varStructSize);
    
    // for (int childID=0;childID<ugg->children.size();childID++) {
    //   UserGeom::SP child = geometry[childID]->as<UserGeom>();
    // assert(child);
    UserGeom::DeviceData &ugDD = getDD(device);
    ugDD.internalBufferForBoundsProgram.alloc(primCount*sizeof(box3f));
      // = ug->internalBufferForBoundsProgram.get();

      writeVariables(userGeomData.data(),device->ID);
      // cb(userGeomData.data(),context->owlDeviceID,
      //    ug->geomID,childID,cbData); 
        
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
        
      void  *d_geomData = tempMem.get();//nullptr;
      vec3f *d_boundsArray = (vec3f*)ugDD.internalBufferForBoundsProgram.get();
      void  *args[] = {
        &d_geomData,
        &d_boundsArray,
        (void *)&primCount
      };
        
      // ll::GeomType *gt = device->checkGetGeomType(geomType->ID);//ug->geomTypeID);
      CUstream stream = device->context->stream;
      UserGeomType::DeviceData &typeDD = getTypeDD(device);
      if (!typeDD.boundsFuncKernel)
        throw std::runtime_error("bounds kernel set, but not yet compiled - did you forget to call BuildPrograms() before (User)GroupAccelBuild()!?");
        
      CUresult rc
        = cuLaunchKernel(typeDD.boundsFuncKernel,
                         gridDims.x,gridDims.y,gridDims.z,
                         blockDims.x,blockDims.y,blockDims.z,
                         0, stream, args, 0);
      if (rc) {
        const char *errName = 0;
        cuGetErrorName(rc,&errName);
        PRINT(errName);
        exit(0);
      }
    // }
    tempMem.free();
    cudaDeviceSynchronize();
    device->popActive(oldActive);
  }

  void UserGeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                            GeomType *_parent,
                                            Device *device,
                                            int rt)
  {
    GeomType::DeviceData::fillPGDesc(pgDesc,_parent,device,rt);
    UserGeomType *parent = (UserGeomType*)_parent;
    
    // ----------- intserect -----------
    if (rt < parent->intersectProg.size()) {
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
    PING;
    if (!boundsProg.module) return;
    Module::SP module = boundsProg.module;
    assert(module);

    for (auto device : context->getDevices()) {
      LOG("building bounds function ....");
      const int oldActive = device->pushActive();
      auto &typeDD = getDD(device);
      auto &moduleDD = module->getDD(device);
      
      assert(moduleDD.boundsModule);

      const std::string annotatedProgName
        = std::string("__boundsFuncKernel__")
        + boundsProg.progName;
    
      CUresult rc = cuModuleGetFunction(&typeDD.boundsFuncKernel,
                                        moduleDD.boundsModule,
                                        annotatedProgName.c_str());
      device->popActive(oldActive);
      
      switch(rc) {
      case CUDA_SUCCESS:
        /* all OK, nothing to do */
        LOG_OK("found bounds function " << annotatedProgName << " ... perfect!");
        break;
      case CUDA_ERROR_NOT_FOUND:
        throw std::runtime_error("in "+std::string(__PRETTY_FUNCTION__)
                                 +": could not find OPTIX_BOUNDS_PROGRAM("
                                 +boundsProg.progName+")");
      default:
        const char *errName = 0;
        cuGetErrorName(rc,&errName);
        PRINT(errName);
        exit(0);
      }
    }
  }

} // ::owl
