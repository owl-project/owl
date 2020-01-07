// ======================================================================== //
// Copyright 2018-2019 Ingo Wald                                            //
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

#include "SampleRenderer.h"
#include "LaunchParams.h"
// this include may only appear in a single source file:
#include <optix_function_table_definition.h>
#include <string.h>

/*! \namespace osc - Optix Siggraph Course */
namespace osc {

  extern "C" char embedded_ptx_code[];

  /*! SBT record for a raygen program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) RaygenRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a miss program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) MissRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    // just a dummy value - later examples will use more interesting
    // data here
    void *data;
  };

  /*! SBT record for a hitgroup program */
  struct __align__( OPTIX_SBT_RECORD_ALIGNMENT ) HitgroupRecord
  {
    __align__( OPTIX_SBT_RECORD_ALIGNMENT ) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    TriangleMeshSBTData data;
  };


  /*! constructor - performs all setup, including initializing
    optix, creates module, pipeline, programs, SBT, etc. */
  SampleRenderer::SampleRenderer(const Model *model, const QuadLight &light)
    : model(model)
  {
    // createContext();
    context = owlContextCreate();
    
    // createModule();
    module = owlModuleCreate(context,embedded_ptx_code);
    // createRaygenPrograms();
    rayGen
      = owlRayGenCreate(context,module,"renderFrame",
                        /* no sbt data: */0,nullptr,-1);
    // createMissPrograms();
    missProgRadiance
      = owlMissProgCreate(context,module,"radiance",
                          /* no sbt data: */0,nullptr,-1);
    missProgShadow
      = owlMissProgCreate(context,module,"shadow",
                          /* no sbt data: */0,nullptr,-1);

    buildAccel();
    
    owlBuildPipeline(context);
    owlBuildPrograms(context);
    owlBuildSBT(context);
    
    OWLVarDecl launchParamsVars[] = {
      { "numPixelSamples", OWL_INT,    OWL_OFFSETOF(LaunchParams,numPixelSamples)},
      { "frameID", OWL_INT,    OWL_OFFSETOF(LaunchParams,frame.frameID)},
      // light settings:
      { "light.origin",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.origin)},
      { "light.du",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.du)},
      { "light.dv",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.dv)},
      { "light.power",    OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,light.power)},
      // camera settings:
      { "camera.position", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.position)},
      { "camera.direction", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.direction)},
      { "camera.horizontal", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.horizontal)},
      { "camera.vertical", OWL_FLOAT3, OWL_OFFSETOF(LaunchParams,camera.vertical)},
      { /* sentinel to mark end of list */ }
    };
    launchParams
      = owlLaunchParamsCreate(context,sizeof(LaunchParams),
                              launchParamsVars,-1);
    owlLaunchParamsSet3f(launchParams,"light.origin",(const owl3f&)light.origin);
    owlLaunchParamsSet3f(launchParams,"light.du",(const owl3f&)light.du);
    owlLaunchParamsSet3f(launchParams,"light.dv",(const owl3f&)light.dv);
    owlLaunchParamsSet3f(launchParams,"light.power",(const owl3f&)light.power);
  }

  void SampleRenderer::createTextures()
  {
    int numTextures = (int)model->textures.size();

    textureArrays.resize(numTextures);
    textureObjects.resize(numTextures);
    
    for (int textureID=0;textureID<numTextures;textureID++) {
      auto texture = model->textures[textureID];
      
      cudaResourceDesc res_desc = {};
      
      cudaChannelFormatDesc channel_desc;
      int32_t width  = texture->resolution.x;
      int32_t height = texture->resolution.y;
      int32_t numComponents = 4;
      int32_t pitch  = width*numComponents*sizeof(uint8_t);
      channel_desc = cudaCreateChannelDesc<uchar4>();
      
      cudaArray_t   &pixelArray = textureArrays[textureID];
      CUDA_CHECK(MallocArray(&pixelArray,
                             &channel_desc,
                             width,height));
      
      CUDA_CHECK(Memcpy2DToArray(pixelArray,
                                 /* offset */0,0,
                                 texture->pixel,
                                 pitch,pitch,height,
                                 cudaMemcpyHostToDevice));
      
      res_desc.resType          = cudaResourceTypeArray;
      res_desc.res.array.array  = pixelArray;
      
      cudaTextureDesc tex_desc     = {};
      tex_desc.addressMode[0]      = cudaAddressModeWrap;
      tex_desc.addressMode[1]      = cudaAddressModeWrap;
      tex_desc.filterMode          = cudaFilterModeLinear;
      tex_desc.readMode            = cudaReadModeNormalizedFloat;
      tex_desc.normalizedCoords    = 1;
      tex_desc.maxAnisotropy       = 1;
      tex_desc.maxMipmapLevelClamp = 99;
      tex_desc.minMipmapLevelClamp = 0;
      tex_desc.mipmapFilterMode    = cudaFilterModePoint;
      tex_desc.borderColor[0]      = 1.0f;
      tex_desc.sRGB                = 0;
      
      // Create texture object
      cudaTextureObject_t cuda_tex = 0;
      CUDA_CHECK(CreateTextureObject(&cuda_tex, &res_desc, &tex_desc, nullptr));
      textureObjects[textureID] = cuda_tex;
    }
  }
  
  void SampleRenderer::buildAccel()
  {
    const int numMeshes = (int)model->meshes.size();
    std::vector<OWLGeom> meshes;

    OWLBuffer texturesBuffer 
      = owlDeviceBufferCreate(context,OWL_USER_TYPE(cudaTextureObject_t),
                              textureObjects.size(),textureObjects.data());
    
    OWLVarDecl triMeshVars[] = {
      { "color",    OWL_FLOAT3, OWL_OFFSETOF(TriangleMeshSBTData,color) },
      { "vertex",   OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,vertex) },
      { "normal",   OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,normal) },
      { "index",    OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,index) },
      { "texcoord", OWL_BUFPTR, OWL_OFFSETOF(TriangleMeshSBTData,texcoord) },
      { "hasTexture",OWL_INT,
        OWL_OFFSETOF(TriangleMeshSBTData,index) },
      { "texture",   OWL_USER_TYPE(cudaTextureObject_t),
        OWL_OFFSETOF(TriangleMeshSBTData,index) },
      { nullptr /* sentinel to mark end of list */ }
    };
    OWLGeomType triMeshGeomType
      = owlGeomTypeCreate(context,
                          OWL_GEOM_TRIANGLES,
                          sizeof(TriangleMeshSBTData),
                          triMeshVars,-1);
    std::vector<OWLGeom> geoms;
    for (int meshID=0;meshID<numMeshes;meshID++) {
      // upload the model to the device: the builder
      TriangleMesh &mesh = *model->meshes[meshID];
      
      OWLBuffer vertexBuffer 
        = owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.vertex.size(),
                                mesh.vertex.data());
      OWLBuffer indexBuffer  
        = owlDeviceBufferCreate(context,OWL_INT3,mesh.index.size(),
                                mesh.index.data());
      OWLBuffer normalBuffer 
        = mesh.normal.empty()
        ? nullptr
        : owlDeviceBufferCreate(context,OWL_FLOAT3,mesh.normal.size(),
                                mesh.normal.data());
      OWLBuffer texcoordBuffer
        = mesh.texcoord.empty()
        ? nullptr
        : owlDeviceBufferCreate(context,OWL_FLOAT2,mesh.texcoord.size(),
                                mesh.texcoord.data());
      // create the geom
      OWLGeom geom
        = owlGeomCreate(context,triMeshGeomType);

      // set the specific vertex/index buffers required to build the accel
      owlTrianglesSetVertices(geom,vertexBuffer,
                              mesh.vertex.size(),sizeof(vec3f),0);
      owlTrianglesSetIndices(geom,indexBuffer,
                             mesh.index.size(),sizeof(vec3i),0);
      // set sbt data
      owlGeomSetBuffer(geom,"index",indexBuffer);
      owlGeomSetBuffer(geom,"vertex",vertexBuffer);
      owlGeomSetBuffer(geom,"normal",normalBuffer);
      owlGeomSetBuffer(geom,"texcoord",texcoordBuffer);

      owlGeomSet3f(geom,"color",(const owl3f &)mesh.diffuse);
      if (mesh.diffuseTextureID >= 0) {
        owlGeomSet1i(geom,"hasTexture",1);
      } else {
        owlGeomSet1i(geom,"hasTexture",0);
      }
      geoms.push_back(geom);
    }
    
    world = owlTrianglesGeomGroupCreate(context,geoms.size(),geoms.data());
  }
  

  /*! render one frame */
  void SampleRenderer::render()
  {
#if 0
    // sanity check: make sure we launch only after first resize is
    // already done:
    if (launchParams.frame.size.x == 0) return;

    if (!accumulate)
      launchParams.frame.frameID = 0;
    launchParamsBuffer.upload(&launchParams,1);
    launchParams.frame.frameID++;
    
    OPTIX_CHECK(optixLaunch(/*! pipeline we're launching launch: */
                            pipeline,stream,
                            /*! parameters and SBT */
                            launchParamsBuffer.d_pointer(),
                            launchParamsBuffer.sizeInBytes,
                            &sbt,
                            /*! dimensions of the launch: */
                            launchParams.frame.size.x,
                            launchParams.frame.size.y,
                            1
                            ));

    OptixDenoiserParams denoiserParams;
    denoiserParams.denoiseAlpha = 1;
    denoiserParams.hdrIntensity = denoiserIntensity.d_pointer();
    denoiserParams.blendFactor  = 1.f/(launchParams.frame.frameID);
    
    // -------------------------------------------------------
    OptixImage2D inputLayer[3];
    inputLayer[0].data = fbColor.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[0].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[0].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[0].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[0].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[0].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[2].data = fbNormal.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[2].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[2].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[2].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[2].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[2].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // ..................................................................
    inputLayer[1].data = fbAlbedo.d_pointer();
    /// Width of the image (in pixels)
    inputLayer[1].width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    inputLayer[1].height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    inputLayer[1].rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    inputLayer[1].pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    inputLayer[1].format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    OptixImage2D outputLayer;
    outputLayer.data = denoisedBuffer.d_pointer();
    /// Width of the image (in pixels)
    outputLayer.width = launchParams.frame.size.x;
    /// Height of the image (in pixels)
    outputLayer.height = launchParams.frame.size.y;
    /// Stride between subsequent rows of the image (in bytes).
    outputLayer.rowStrideInBytes = launchParams.frame.size.x * sizeof(float4);
    /// Stride between subsequent pixels of the image (in bytes).
    /// For now, only 0 or the value that corresponds to a dense packing of pixels (no gaps) is supported.
    outputLayer.pixelStrideInBytes = sizeof(float4);
    /// Pixel format.
    outputLayer.format = OPTIX_PIXEL_FORMAT_FLOAT4;

    // -------------------------------------------------------
    if (denoiserOn) {
      OPTIX_CHECK(optixDenoiserComputeIntensity
                  (denoiser,
                   /*stream*/0,
                   &inputLayer[0],
                   (CUdeviceptr)denoiserIntensity.d_pointer(),
                   (CUdeviceptr)denoiserScratch.d_pointer(),
                   denoiserScratch.size()));
      
      OPTIX_CHECK(optixDenoiserInvoke(denoiser,
                                      /*stream*/0,
                                      &denoiserParams,
                                      denoiserState.d_pointer(),
                                      denoiserState.size(),
                                      &inputLayer[0],2,
                                      /*inputOffsetX*/0,
                                      /*inputOffsetY*/0,
                                      &outputLayer,
                                      denoiserScratch.d_pointer(),
                                      denoiserScratch.size()));
    } else {
      cudaMemcpy((void*)outputLayer.data,(void*)inputLayer[0].data,
                 outputLayer.width*outputLayer.height*sizeof(float4),
                 cudaMemcpyDeviceToDevice);
    }
    computeFinalPixelColors();
#endif
    
    // sync - make sure the frame is rendered before we download and
    // display (obviously, for a high-performance application you
    // want to use streams and double-buffering, but for this simple
    // example, this will have to do)
    CUDA_SYNC_CHECK();
  }

  /*! set camera to render with */
  void SampleRenderer::setCamera(const Camera &camera)
  {
    lastSetCamera = camera;
    // reset accumulation
    // launchParams.frame.frameID = 0;
    frameID = 0;
    owlLaunchParamsSet1i(launchParams,"frameID",0);
    const vec3f position  = camera.from;
    const vec3f direction = normalize(camera.at-camera.from);
    
    const float cosFovy = 0.66f;
    const float aspect
      = float(fbSize.x)
      / float(fbSize.y);
    const vec3f horizontal
      = cosFovy * aspect * normalize(cross(direction,
                                           camera.up));
    const vec3f vertical
      = cosFovy * normalize(cross(horizontal,
                                  direction));
    owlLaunchParamsSet3f(launchParams,"camera.position",(const owl3f&)position);
    owlLaunchParamsSet3f(launchParams,"camera.direction",(const owl3f&)direction);
    owlLaunchParamsSet3f(launchParams,"camera.vertical",(const owl3f&)vertical);
    owlLaunchParamsSet3f(launchParams,"camera.horizontal",(const owl3f&)horizontal);
  }
  
  /*! resize frame buffer to given resolution */
  void SampleRenderer::resize(const vec2i &newSize)
  {
    // if (denoiser) {
    //   OPTIX_CHECK(optixDenoiserDestroy(denoiser));
    // };
    if (fbColor) {
      std::cout << "todo: buffer destroy" << std::endl;
      // owlBufferDestroy(fbColor);
    }

    this->fbSize = newSize;
    // fbColor = owlDeviceBufferCreate(context,OWL_FLOAT4,fbSize.x*fbSize.y,nullptr);
    fbColor = owlHostPinnedBufferCreate(context,OWL_INT,fbSize.x*fbSize.y);
    // fbColor = owlDeviceBufferCreate(context,OWL_FLOAT4,fbSize.x*fbSize.y,nullptr);

    owlLaunchParamsSetBuffer(launchParams,"frame.colorBuffer",fbColor);
    owlLaunchParamsSet2i(launchParams,"frame.fbSize",(const owl2i&)fbSize);

    // and re-set the camera, since aspect may have changed
    setCamera(lastSetCamera);

    // // ------------------------------------------------------------------
    // OPTIX_CHECK(optixDenoiserSetup(denoiser,0,
    //                                newSize.x,newSize.y,
    //                                denoiserState.d_pointer(),
    //                                denoiserState.size(),
    //                                denoiserScratch.d_pointer(),
    //                                denoiserScratch.size()));
  }
  
  /*! download the rendered color buffer */
  void SampleRenderer::downloadPixels(uint32_t h_pixels[])
  {
    std::cout <<" todo: download buffer..." << std::endl;
    memcpy(h_pixels,owlBufferGetPointer(fbColor,0),fbSize.x*fbSize.y*sizeof(int));
    // finalColorBuffer.download(h_pixels,
    //                           launchParams.frame.size.x*launchParams.frame.size.y);
  }
  
} // ::osc
