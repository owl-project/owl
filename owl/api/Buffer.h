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

#include "RegisteredObject.h"
#include "ll/Device.h"
#include "Texture.h"

namespace owl {

  struct Buffer : public RegisteredObject
  {
    typedef std::shared_ptr<Buffer> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public RegisteredObject::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      DeviceData(const DeviceContext::SP &device)
        : RegisteredObject::DeviceData(device)
      {};
      
      void *d_pointer { 0 };
    };
    
    Buffer(Context *const context,
           OWLDataType type);
    
    /*! destructor - free device data, de-regsiter, and destruct */
    virtual ~Buffer();
    
    std::string toString() const override { return "Buffer"; }


    Buffer::DeviceData &getDD(const DeviceContext::SP &device) const
    {
      assert(device);
      assert(device->ID < deviceData.size());
      return *deviceData[device->ID]->as<Buffer::DeviceData>();
    }
      
    // const void *getPointer(int deviceID) const
    // { std::cout << "deprecated - kill this" << std::endl; return getDD(deviceID).d_pointer; }
    const void *getPointer(const DeviceContext::SP &device) const
    { assert(device); return getDD(device).d_pointer; }
        
    size_t getElementCount() const;
    
    virtual void resize(size_t newElementCount) = 0;
    virtual void upload(const void *hostPtr) = 0;
    virtual void upload(const int deviceID, const void *hostPtr) = 0;
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    /*! destroy whatever resouces this buffer's ll-layer handle this
        may refer to; this will not destruct the current object
        itself, but should already release all its references */
    void destroy();

    size_t sizeInBytes() const { return elementCount * sizeOf(type); }
    
    const OWLDataType type;
    size_t      elementCount { 0 };
  };

  struct DeviceBuffer : public Buffer {
    typedef std::shared_ptr<DeviceBuffer> SP;

    /*! any device-specific data, such as optix handles, cuda device
        pointers, etc */
    struct DeviceData : public Buffer::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      DeviceData(DeviceBuffer *parent, const DeviceContext::SP &device)
        : Buffer::DeviceData(device), parent(parent), device(device)
      {}
      virtual ~DeviceData();

      /*! executes the resize on the given device, including freeing
          old memory, and allocating required elemnts in device
          format, as required */
      virtual void executeResize() = 0;
      
      /*! create an async upload for data from the given host data
          pointer, using the given device's cuda stream, and doing any
          required translation from host-side data (eg, a texture
          object handle) to device-side data (ie, the
          cudaTextreObject_t for that device). this will *not* wait
          for the upload to complete, so an explicit cuda sync has to
          be done to ensure no race conditiosn will occur */
      virtual void uploadAsync(const void *hostDataPtr) = 0;

      DeviceContext::SP const device;
      DeviceBuffer *const parent;
    };
    struct DeviceDataForTextures : public DeviceData {
      DeviceDataForTextures(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr) override;
    
      /*! this is used only for buffers over object types (bufers of
        textures, or buffers of buffers). For those buffers, we use this
        vector to store host-side handles of the objects in this buffer,
        to ensure proper recounting */
      std::vector<Texture::SP> hostHandles;
    };
    struct DeviceDataForBuffers : public DeviceData {
      DeviceDataForBuffers(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr) override;
    
      /*! this is used only for buffers over object types (bufers of
        textures, or buffers of buffers). For those buffers, we use this
        vector to store host-side handles of the objects in this buffer,
        to ensure proper recounting */
      std::vector<Buffer::SP> hostHandles;
    };
    struct DeviceDataForCopyableData : public DeviceData {
      DeviceDataForCopyableData(DeviceBuffer *parent, const DeviceContext::SP &device)
        : DeviceData(parent,device)
      {}
      void executeResize() override;
      void uploadAsync(const void *hostDataPtr) override;
    };
    
    DeviceBuffer(Context *const context,
                 OWLDataType type
                 // ,
                 // size_t count
                 // ,
                 // const void *init
                 );

    /*! pretty-printer, for debugging */
    std::string toString() const override { return "DeviceBuffer"; }

    DeviceData &getDD(const DeviceContext::SP &device) const
    {
      assert(device->ID < deviceData.size());
      return *deviceData[device->ID]->as<DeviceData>();
    }
      
    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr) override;
    void upload(const int deviceID, const void *hostPtr) override;
    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
  };
  
  struct HostPinnedBuffer : public Buffer {
    typedef std::shared_ptr<HostPinnedBuffer> SP;
    
    HostPinnedBuffer(Context *const context,
                     OWLDataType type// ,
                     // size_t count
                     );

    /*! pretty-printer, for debugging */
    std::string toString() const override { return "HostPinnedBuffer"; }

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr) override;
    void upload(const int deviceID, const void *hostPtr) override;

    void *cudaHostPinnedMem { 0 };
  };
  
  struct ManagedMemoryBuffer : public Buffer {
    typedef std::shared_ptr<ManagedMemoryBuffer> SP;
    
    ManagedMemoryBuffer(Context *const context,
                        OWLDataType type// ,
                        // size_t count
                        // ,
                        // /*! data with which to populate this buffer;
                        //   may be null, but has to be of size 'amount'
                        //   if not */
                        // const void *initData
                        );

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr) override;
    void upload(const int deviceID, const void *hostPtr) override;
    /*! creates the device-specific data for this group */
    // Buffer::DeviceData::SP createOn(ll::const DeviceContext::SP &device) override;


    /*! pretty-printer, for debugging */
    std::string toString() const override { return "ManagedMemoryBuffer"; }

    void *cudaManagedMem { 0 };
  };

  struct GraphicsBuffer : public Buffer {
    typedef std::shared_ptr<GraphicsBuffer> SP;

    // /*! any device-specific data, such as optix handles, cuda device
    //     pointers, etc */
    // struct DeviceData : public Buffer::DeviceData {
    //   typedef std::shared_ptr<DeviceData> SP;
    // };

    GraphicsBuffer(Context* const context,
                   OWLDataType type,
                   // size_t count,
                   cudaGraphicsResource_t resource);

    void map(const int deviceID=0, CUstream stream=0);
    void unmap(const int deviceID=0, CUstream stream=0);

    void resize(size_t newElementCount) override;
    void upload(const void *hostPtr) override;
    void upload(const int deviceID, const void *hostPtr) override;
    /*! creates the device-specific data for this group */
    // Buffer::DeviceData::SP createOn(ll::const DeviceContext::SP &device) override;

    /*! the cuda graphics resource to map to - note that this is
        probably valid on only one GPU */
    cudaGraphicsResource_t resource;
    
    /*! pretty-printer, for debugging */
    std::string toString() const override { return "GraphicsBuffer"; }
  };
  
} // ::owl
