// ======================================================================== //
// Copyright 2020 Ingo Wald                                                 //
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

#include "Texture.h"
#include "Context.h"
#include "owl/ll/Device.h"
#include "owl/ll/helper/cuda.h"

namespace owl {

  Texture::Texture(Context *const context,
                   vec2i                size,
                   uint32_t             linePitchInBytes,
                   OWLTexelFormat       texelFormat,
                   OWLTextureFilterMode filterMode,
                   const void *texels
                   )
    : RegisteredObject(context,context->textures)
  {
    assert(size.x > 0);
    assert(size.y > 0);
    int32_t pitch  = linePitchInBytes;
    
    assert(
      (texelFormat == OWL_TEXEL_FORMAT_RGBA8) ||
      (texelFormat == OWL_TEXEL_FORMAT_RGBA32F) ||
      (texelFormat == OWL_TEXEL_FORMAT_R8) ||
      (texelFormat == OWL_TEXEL_FORMAT_R32F)
    );
    if (pitch == 0) {
      switch(texelFormat) {
        case OWL_TEXEL_FORMAT_RGBA8:   pitch = size.x*sizeof(vec4uc); break;
        case OWL_TEXEL_FORMAT_RGBA32F: pitch = size.x*sizeof(vec4f); break;
        case OWL_TEXEL_FORMAT_R8:      pitch = size.x*sizeof(uint8_t); break;
        case OWL_TEXEL_FORMAT_R32F:    pitch = size.x*sizeof(float); break;
        default: assert(false);
      }  
    }

    assert(texels != nullptr);
    
    for (auto device : context->llo->devices) {
      int oldActive = device->context->pushActive();

      cudaResourceDesc res_desc = {};
      
      cudaChannelFormatDesc channel_desc;
      switch(texelFormat) {
        case OWL_TEXEL_FORMAT_RGBA8:   channel_desc = cudaCreateChannelDesc<uchar4>(); break;
        case OWL_TEXEL_FORMAT_RGBA32F: channel_desc = cudaCreateChannelDesc<float4>(); break;
        case OWL_TEXEL_FORMAT_R8:      channel_desc = cudaCreateChannelDesc<uint8_t>(); break;
        case OWL_TEXEL_FORMAT_R32F:    channel_desc = cudaCreateChannelDesc<float>(); break;
        default: assert(false);
      }        

      cudaArray_t   pixelArray;
      CUDA_CALL(MallocArray(&pixelArray,
                             &channel_desc,
                             size.x,size.y));
      textureArrays.push_back(pixelArray);
      
      CUDA_CALL(Memcpy2DToArray(pixelArray,
                                 /* offset */0,0,
                                 texels,
                                 pitch,pitch,size.y,
                                 cudaMemcpyHostToDevice));
      
      res_desc.resType          = cudaResourceTypeArray;
      res_desc.res.array.array  = pixelArray;
      
      cudaTextureDesc tex_desc     = {};
      tex_desc.addressMode[0]      = cudaAddressModeClamp;
      tex_desc.addressMode[1]      = cudaAddressModeClamp;
      assert(filterMode == OWL_TEXTURE_NEAREST
             ||
             filterMode == OWL_TEXTURE_LINEAR);
      tex_desc.filterMode          =
        filterMode == OWL_TEXTURE_NEAREST
        ? cudaFilterModePoint
        : cudaFilterModeLinear;
      tex_desc.readMode            =
        ((texelFormat == OWL_TEXEL_FORMAT_R8) || (texelFormat == OWL_TEXEL_FORMAT_RGBA8)) ?
        cudaReadModeNormalizedFloat : cudaReadModeElementType;
      tex_desc.normalizedCoords    = 1;
      tex_desc.maxAnisotropy       = 1;
      tex_desc.maxMipmapLevelClamp = 99;
      tex_desc.minMipmapLevelClamp = 0;
      tex_desc.mipmapFilterMode    = cudaFilterModePoint;
      tex_desc.borderColor[0]      = 1.0f;
      tex_desc.sRGB                = 0;
      
      // Create texture object
      cudaTextureObject_t cuda_tex = 0;
      CUDA_CALL(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));

      textureObjects.push_back(cuda_tex);
      
      device->context->popActive(oldActive);
    }
  }

  /* return the cuda texture object corresponding to the specified 
       device ID*/
  cudaTextureObject_t Texture::getObject(int deviceID)
  {
    return textureObjects[deviceID];
  }

  Texture::~Texture()
  {
    PING;
    std::cout << "NOT IMPLEMENTED" << std::endl;
    exit(0);
  }
  
} // ::owl
