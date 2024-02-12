// ======================================================================== //
// Copyright 2020-2021 Ingo Wald                                            //
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

#include "pyOWL/Context.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"

namespace pyOWL {

  Context::Context()
  {
    std::cout << OWL_TERMINAL_LIGHT_GREEN
              << "#pyOWL: creating context..."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    
    handle = owlContextCreate(nullptr,1);
    std::cout << OWL_TERMINAL_GREEN
              << "#pyOWL: context created."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
  }
    
  Context::~Context()
  {
    destroy();
  }

  std::shared_ptr<Module>
  Context::createModuleFromFile(const std::string &fileName)
  {
    return Module::fromFile(shared_from_this(),fileName);
  }
  std::shared_ptr<Module>
  Context::createModuleFromString(const std::string &ptx)
  {
    return Module::fromString(shared_from_this(),ptx);
  }

  std::shared_ptr<GeomType>
  Context::createGeomType(const int kind,
                          const std::shared_ptr<Module> &module,
                          const std::string &name)
  {
    return std::make_shared<GeomType>(this,(OWLGeomKind)kind,module,name);
  }
    
  void Context::buildPrograms()
  {
    owlBuildPrograms(handle);
  }
  
  void Context::buildPipeline()
  {
    owlBuildPipeline(handle);
  }
  
  void Context::buildSBT()
  {
    owlBuildSBT(handle);
  }


  std::shared_ptr<Context> createContext()
  {
    return std::make_shared<Context>();
  }
  
  Buffer::SP Context::createDeviceBuffer(int type, const py::buffer &buffer)
  {
    return DeviceBuffer::create(this,(OWLDataType)type,buffer);
  }
  
  Buffer::SP Context::createHostPinnedBuffer(int type, int size)
  {
    return HostPinnedBuffer::create(this,(OWLDataType)type,size);
    // return Buffer::create(this,(OWLDataType)type,buffer);
  }

  Geom::SP Context::createGeom(const std::shared_ptr<GeomType> type)
  {
    return Geom::create(this,type);
  }

  MissProg::SP Context::createMissProg(const std::shared_ptr<Module> &module,
                                       const std::string &typeName,
                                       const std::string &funcName)
  {
    return MissProg::create(this,module,typeName,funcName);
  }

  RayGen::SP Context::createRayGen(const std::shared_ptr<Module> &module,
                                   const std::string &typeName,
                                   const std::string &funcName)
  {
    return RayGen::create(this,module,typeName,funcName);
  }

  /*! allows to query whether the user has already explicitly called
    contextDestroy. if so, any releases of handles are no longer
    valid because whatever they may have pointed to inside the
    (owl-)context is already dead */
  bool Context::alive()
  {
    return handle != 0;
  }
  
  void Context::destroy()
  {
    if (!handle)
      // already destroyed, probably becasue the user called an explicit context::destroy()
      return;
    
    std::cout << OWL_TERMINAL_GREEN
              << "#pyOWL: context shutting down."
              << OWL_TERMINAL_DEFAULT
              << std::endl;
    owlContextDestroy(handle);
    handle = 0;
  }
  
  void save_png_rgba8(Buffer::SP buffer,
                      const std::vector<int> &_fbSize,
                      const std::string &outFileName)
  {
    assert(buffer);
    assert(buffer->getHandle());
    
    vec2i fbSize = make_vec2i(_fbSize);
    const uint32_t *fb
      = (const uint32_t*)owlBufferGetPointer(buffer->getHandle(),0);
    assert(fb);
    
    stbi_write_png(outFileName.c_str(),fbSize.x,fbSize.y,4,
                   fb,fbSize.x*sizeof(uint32_t));
  }

  // Group::SP Context::createTrianglesGeomGroup(const std::vector<Geom::SP> &list
  //                                             //const py::list &list
  //                                             )
  // {
  //   // return Group::createTrianglesGG(this,list);
  // }

}
