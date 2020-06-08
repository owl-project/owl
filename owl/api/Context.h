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

#pragma once

#include "ObjectRegistry.h"
#include "Buffer.h"
#include "Texture.h"
#include "Group.h"
#include "RayGen.h"
#include "LaunchParams.h"
#include "MissProg.h"
// ll
#include "../ll/DeviceGroup.h"

namespace owl {

  std::string typeToString(const OWLDataType type);
  
  struct Context : public Object {
    typedef std::shared_ptr<Context> SP;

    static Context::SP create(int32_t *requestedDeviceIDs,
                              int      numRequestedDevices);

    Context(int32_t *requestedDeviceIDs,
            int      numRequestedDevices);
    
    virtual ~Context();

    ObjectRegistryT<Buffer>       buffers;
    ObjectRegistryT<Texture>      textures;
    ObjectRegistryT<Group>        groups;
    ObjectRegistryT<RayGenType>   rayGenTypes;
    ObjectRegistryT<RayGen>       rayGens;
    ObjectRegistryT<MissProgType> missProgTypes;
    ObjectRegistryT<MissProg>     missProgs;
    ObjectRegistryT<GeomType>     geomTypes;
    ObjectRegistryT<Geom>         geoms;
    ObjectRegistryT<Module>       modules;
    ObjectRegistryT<LaunchParamsType> launchParamTypes;
    ObjectRegistryT<LaunchParams>     launchParams;
    
    //! TODO: allow changing that via api ..
    size_t numRayTypes = 1;

    void setRayTypeCount(size_t rayTypeCount);

    /*! sets maximum instancing depth for the given context:

      '0' means 'no instancing allowed, only bottom-level accels; 
  
      '1' means 'at most one layer of instances' (ie, a two-level scene),
      where the 'root' world rays are traced against can be an instance
      group, but every child in that inscne group is a geometry group.

      'N>1" means "up to N layers of instances are allowed.

      The default instancing depth is 1 (ie, a two-level scene), since
      this allows for most use cases of instancing and is still
      hardware-accelerated. Using a node graph with instancing deeper than
      the configured value will result in wrong results; but be aware that
      using any value > 1 here will come with a cost. It is recommended
      to, if at all possible, leave this value to one and convert the
      input scene to a two-level scene layout (ie, with only one level of
      instances) */
    void setMaxInstancingDepth(int32_t maxInstanceDepth);

  /*! experimentation code for sbt construction */
    void buildSBT();
    void buildPipeline();
    void buildPrograms();
    
    // InstanceGroup::SP
    // createInstanceGroup(size_t numChildren,
    //                     Group::SP      *groups,
    //                     const uint32_t *instIDs,
    //                     const float    *xfms,
    //                     OWLMatrixFormat matrixFormat)
    // // InstanceGroup::SP createInstanceGroup(size_t numChildren)
    // {
    //   return std::make_shared<InstanceGroup>
    //     (this,numChildren,groups,instIDs,xfms,matrixFormat);
    // }

    
    GeomGroup::SP
    trianglesGeomGroupCreate(size_t numChildren);
    
    GeomGroup::SP
    userGeomGroupCreate(size_t numChildren);
    
    Buffer::SP
    deviceBufferCreate(OWLDataType type,
                       size_t count,
                       const void *init);

    Texture::SP
    texture2DCreate(OWLTexelFormat texelFormat,
                    OWLTextureFilterMode filterMode,
                    const vec2i size,
                    uint32_t linePitchInBytes,
                    const void *texels);
    
    /*! creates a buffer that uses CUDA host pinned memory; that
      memory is pinned on the host and accessive to all devices in the
      device group */
    Buffer::SP
    hostPinnedBufferCreate(OWLDataType type,
                           size_t count);
    
    /*! creates a buffer that uses CUDA managed memory; that memory is
      managed by CUDA (see CUDAs documentation on managed memory) and
      accessive to all devices in the deviec group */
    Buffer::SP
    managedMemoryBufferCreate(OWLDataType type,
                              size_t count,
                              const void *init);

    /*! creates a buffer that wraps a CUDA graphics resource
      that can be, for instance, an OpenGL texture */
    Buffer::SP
    graphicsBufferCreate(OWLDataType type,
                         size_t count,
                         cudaGraphicsResource_t resource);
    
    RayGen::SP
    createRayGen(const std::shared_ptr<RayGenType> &type);
    
    RayGenType::SP
    createRayGenType(Module::SP module,
                     const std::string &progName,
                     size_t varStructSize,
                     const std::vector<OWLVarDecl> &varDecls);
    
    LaunchParams::SP
    createLaunchParams(const std::shared_ptr<LaunchParamsType> &type);
    
    LaunchParamsType::SP
    createLaunchParamsType(size_t varStructSize,
                           const std::vector<OWLVarDecl> &varDecls);
    
    MissProg::SP
    createMissProg(const std::shared_ptr<MissProgType> &type);
    
    MissProgType::SP
    createMissProgType(Module::SP module,
                       const std::string &progName,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);
    
    GeomType::SP
    createGeomType(OWLGeomKind kind,
                       size_t varStructSize,
                       const std::vector<OWLVarDecl> &varDecls);
    
    Module::SP createModule(const std::string &ptxCode);

    owl::ll::DeviceGroup *llo;
  };

} // ::owl

