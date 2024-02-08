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

#include "CurvesGeom.h"
#include "Context.h"

namespace owl {

  // ------------------------------------------------------------------
  // CurvesGeomType
  // ------------------------------------------------------------------
  
  /*! construct a new device-data for this type */
  CurvesGeomType::DeviceData::DeviceData(const DeviceContext::SP &device)
    : GeomType::DeviceData(device)
  {}

  CurvesGeomType::CurvesGeomType(Context *const context,
                                 size_t varStructSize,
                                 const std::vector<OWLVarDecl> &varDecls)
    : GeomType(context,varStructSize,varDecls)
  {}

  std::shared_ptr<Geom> CurvesGeomType::createGeom()
  {
    GeomType::SP self
      = std::dynamic_pointer_cast<GeomType>(shared_from_this());
    Geom::SP geom = std::make_shared<CurvesGeom>(context,self);
    geom->createDeviceData(context->getDevices());
    return geom;
  }

  /*! fill in an OptixProgramGroup descriptor with the module and
    program names for this type */
  void CurvesGeomType::DeviceData::fillPGDesc(OptixProgramGroupDesc &pgDesc,
                                              GeomType *_parent,
                                              int rt)
  {
    GeomType::DeviceData::fillPGDesc(pgDesc,_parent,rt);
    CurvesGeomType *parent = (CurvesGeomType*)_parent;


    // ----------- intersect from builtin module -----------
    pgDesc.hitgroup.moduleIS = device->curvesModule[parent->forceCaps][parent->degree-1];
    pgDesc.hitgroup.entryFunctionNameIS = /* default for built-ins */0;
  }
  
  
  // ------------------------------------------------------------------
  // CurvesGeom::DeviceData
  // ------------------------------------------------------------------
  
  CurvesGeom::DeviceData::DeviceData(const DeviceContext::SP &device)
    : Geom::DeviceData(device)
  {}
  
  
  // ------------------------------------------------------------------
  // CurvesGeom
  // ------------------------------------------------------------------
  
  CurvesGeom::CurvesGeom(Context *const context,
                               GeomType::SP geometryType)
    : Geom(context,geometryType)
  {}
  
  /*! pretty-print */
  std::string CurvesGeom::toString() const
  {
    return "CurvesGeom";
  }

  void CurvesGeomType::setDegree(int degree, bool force_caps)
  {
    if (degree < 1 || degree > 3) OWL_RAISE("invalid curve degree (must be 1-3)");
    this->degree = degree;
    this->forceCaps = force_caps;
  }

  /*! set the vertex array (if vector size is 1), or set/enable
    motion blur via multiple time steps, if vector size >= 0 */
  void CurvesGeom::setVertices(const std::vector<Buffer::SP> &vertices,
                               const std::vector<Buffer::SP> &widths,
                               /*! the number of vertices in each time step */
                               size_t count)
  {
    assert(count > 1);
    assert(vertices.size() > 0);
    assert(widths.size() == vertices.size());
    
    vertexCount     = (int)count;
    verticesBuffers = vertices;
    widthsBuffers   = widths;

    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.verticesPointers.clear();
      for (auto va : verticesBuffers)
        dd.verticesPointers.push_back((CUdeviceptr)va->getPointer(device));

      dd.widthsPointers.clear();
      for (auto va : widthsBuffers)
        dd.widthsPointers.push_back((CUdeviceptr)va->getPointer(device));
    }
  }
  
  void CurvesGeom::setSegmentIndices(Buffer::SP indices,
                                     size_t count)
  {
    assert(count > 0);
    assert(indices);
    
    segmentIndicesCount  = (int)count;
    // check for overflow
    assert((size_t)segmentIndicesCount == count);
    segmentIndicesBuffer = indices;
    
    for (auto device : context->getDevices()) {
      DeviceData &dd = getDD(device);
      dd.indicesPointer = (CUdeviceptr)indices->getPointer(device);
    }
  }

    /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  CurvesGeom::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }

    /*! creates the device-specific data for this group */
  RegisteredObject::DeviceData::SP
  CurvesGeomType::createOn(const DeviceContext::SP &device) 
  {
    return std::make_shared<DeviceData>(device);
  }


} // ::owl
