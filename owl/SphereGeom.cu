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

#include "SphereGeom.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // SphereGeomType
  // ------------------------------------------------------------------
  
  /*! construct a new device-data for this type */
  SphereGeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : GeomType::DeviceData(device)
  {}

  SphereGeomType::SphereGeomType(Context *const context,
                                 size_t varStructSize,
                                 const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {}

  std::shared_ptr<Geom> SphereGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<SphereGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  /*! fill in an OptixProgramGroup descriptor with the module and
    program names for this type */
  void SphereGeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                              GeomType *_parent,
                                              int rt)
  {
    GeomType::DeviceData::fillPGDesc(pgDesc,_parent,rt);

    // ----------- intersect from builtin module -----------
    pgDesc.hitgroup.moduleIS = device->spheresModule;
    pgDesc.hitgroup.entryFunctionNameIS = /* default for built-ins */0;
  }
  
  
  // ------------------------------------------------------------------
  // SphereGeom::DeviceData
  // ------------------------------------------------------------------
  
  SphereGeom::DeviceData::DeviceData(const DeviceContext::SP &device)
    : Geom::DeviceData(device)
  {}
  
  
  // ------------------------------------------------------------------
  // SphereGeom
  // ------------------------------------------------------------------
  
  SphereGeom::SphereGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
  {}
  
  /*! pretty-print */
  std::string SphereGeom::toString() const
  {
    return "SphereGeom";
  }

  /*! set the vertex array (if vector size is 1), or set/enable
    motion blur via multiple time steps, if vector size >= 0 */
  void SphereGeom::setVertices(const std::vector<Buffer::SP> &vertices,
                               const std::vector<Buffer::SP> &radii,
                               /*! the number of vertices in each time step */
                               size_t count)
  {
    assert(count > 1);
    assert(vertices.size() > 0);
    assert(radii.size() == vertices.size());
    
    vertexCount = count;
    verticesBuffers = vertices;
    radiusBuffers   = radii;

    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.verticesPointers.clear();
      for (auto va : verticesBuffers)
        dd.verticesPointers.push_back((CUdeviceptr)va->getPointer(device));

      dd.radiusPointers.clear();
      for (auto va : radiusBuffers)
        dd.radiusPointers.push_back((CUdeviceptr)va->getPointer(device));
    }
  }

  /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  SphereGeom::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

    /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  SphereGeomType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }


} // ::owl
