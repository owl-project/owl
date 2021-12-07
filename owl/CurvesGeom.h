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

#pragma once

#include "Geometry.h"

namespace owl {

  /*! a geometry *type* that uses optix 'curves' primitives, and that
    captures the anyhit and closesthit programs, variable types, SBT
    layout, etc, associated with all instances of this type */
  struct CurvesGeomType : public GeomType {
    typedef std::shared_ptr<CurvesGeomType> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public GeomType::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! construct a new device-data for this type */
      DeviceData(const DeviceContext::SP &device);

      /*! fill in an OptixProgramGroup descriptor with the module and
        program names for this type; this uses the parent class to fill
        in CH and AH programs, but sets IS program to optix's builtin
        curves intersector */
      void fillPGDesc(OptixProgramGroupDesc &pgDesc,
                      GeomType *gt,
                      int rayType) override;
    };
    
    CurvesGeomType(Context *const context,
                   size_t varStructSize,
                   const std::vector<OWLVarDecl> &varDecls);
    
    /*! pretty-print */
    std::string toString() const override { return "CurvesGeomType"; }
    
    std::shared_ptr<Geom> createGeom() override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
    
    /*! create this object's device-specific data for the device */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;

    void setDegree(int degree, bool force_caps);
    
    /*! curve degree - linear (1), quadratic (2), or cubic (3) */
    int degree = 3;
    
    /*! if true, we tell optix to add curve caps on qudratc and cubic
        segments (which by default wll not have caps). Ignored for
        linear segments, which always have caps */
    bool forceCaps = false;
   };

  /*! an actual *instance* of a given curves primitives; this geometry
    object captures the programs and SBT data associated with its
    associated CurvesGeomType, and will "instantiate" these with the
    right control points (vertices and vertex widths'), segment
    indices, degree, etc */
  struct CurvesGeom : public Geom {
    
    typedef std::shared_ptr<CurvesGeom> SP;

    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public Geom::DeviceData {
      DeviceData(const DeviceContext::SP &device);

      /*! this is a *vector* of vertex arrays, for motion blur
        purposes. ie, for static meshes only one entry is used, for
        motion blur two (and eventually, maybe more) will be used */
      std::vector<CUdeviceptr> verticesPointers;

      /*! this is a *vector* of vertex arrays, for motion blur
        purposes. ie, for static meshes only one entry is used, for
        motion blur two (and eventually, maybe more) will be used */
      std::vector<CUdeviceptr> widthsPointers;

      /*! device poiner to array of segment indices - the memory for the
        indices will live in some sort of buffer; this only points
        to that buffer */
      CUdeviceptr indicesPointer  = (CUdeviceptr)0;
    };

    /*! constructor - create a new (as yet without vertices, indices,
      etc) instance of given curves geom type */
    CurvesGeom(Context *const context,
               GeomType::SP geometryType);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    /*! creates the device-specific data for this group */

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device);
                        
    /*! get reference to the device-specific data for this object's *type* descriptor */
    CurvesGeomType::DeviceData &getTypeDD(const DeviceContext::SP &device) const;

    /*! set the vertex array (if vector size is 1), or set/enable
      motion blur via multiple time steps, if vector size >= 0 */
    void setVertices(const std::vector<Buffer::SP> &vertices,
                     const std::vector<Buffer::SP> &widths,
                     /*! the number of vertices in each time step */
                     size_t count);

    /*! set the index buffer; this remains one buffer even if motion blur is enabled. */
    void setSegmentIndices(Buffer::SP indices,
                           size_t count);

    /*! pretty-print */
    std::string toString() const override;

    int segmentIndicesCount = 0;
    Buffer::SP segmentIndicesBuffer;
    
    int vertexCount = 0;
    std::vector<Buffer::SP> verticesBuffers;
    std::vector<Buffer::SP> widthsBuffers;
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline CurvesGeomType::DeviceData &
  CurvesGeomType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<CurvesGeomType::DeviceData>();
  }

  /*! get reference to given device-specific data for this object */
  inline CurvesGeom::DeviceData &
  CurvesGeom::getDD(const DeviceContext::SP &device)
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<CurvesGeom::DeviceData>();
  }

  /*! get reference to the device-specific data for this object's *type* descriptor */
  inline CurvesGeomType::DeviceData &
  CurvesGeom::getTypeDD(const DeviceContext::SP &device) const
  {
    return (CurvesGeomType::DeviceData &)type->getDD(device);
  }
  
} // ::owl
