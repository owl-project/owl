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

  /*! a geometry *type* that uses optix 'sphere' primitives, and that
    captures the anyhit and closesthit programs, variable types, SBT
    layout, etc, associated with all instances of this type */
  struct SphereGeomType : public GeomType {
    typedef std::shared_ptr<SphereGeomType> SP;
    
    /*! any device-specific data, such as optix handles, cuda device
      pointers, etc */
    struct DeviceData : public GeomType::DeviceData {
      typedef std::shared_ptr<DeviceData> SP;

      /*! construct a new device-data for this type */
      DeviceData(const DeviceContext::SP &device);

      /*! fill in an OptixProgramGroup descriptor with the module and
        program names for this type; this uses the parent class to fill
        in CH and AH programs, but sets IS program to optix's builtin
        sphere intersector */
      void fillPGDesc(OptixProgramGroupDesc &pgDesc,
                      GeomType *gt,
                      int rayType) override;
    };
    
    SphereGeomType(Context *const context,
                   size_t varStructSize,
                   const std::vector<OWLVarDecl> &varDecls);
    
    /*! pretty-print */
    std::string toString() const override { return "SphereGeomType"; }
    
    std::shared_ptr<Geom> createGeom() override;
    
    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device) const;
    
    /*! create this object's device-specific data for the device */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
   };

  /*! an actual *instance* of a given curves primitives; this geometry
    object captures the programs and SBT data associated with its
    associated SphereGeomType, and will "instantiate" these with the
    right control points (vertices and vertex widths'), segment
    indices, degree, etc */
  struct SphereGeom : public Geom {
    
    typedef std::shared_ptr<SphereGeom> SP;

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
      std::vector<CUdeviceptr> radiusPointers;
    };

    /*! constructor - create a new (as yet without vertices, indices,
      etc) instance of given curves geom type */
    SphereGeom(Context *const context,
               GeomType::SP geometryType);

    /*! creates the device-specific data for this group */
    RegisteredObject::DeviceData::SP createOn(const DeviceContext::SP &device) override;
    /*! creates the device-specific data for this group */

    /*! get reference to given device-specific data for this object */
    inline DeviceData &getDD(const DeviceContext::SP &device);
                        
    /*! get reference to the device-specific data for this object's *type* descriptor */
    SphereGeomType::DeviceData &getTypeDD(const DeviceContext::SP &device) const;

    /*! set the vertex array (if vector size is 1), or set/enable
      motion blur via multiple time steps, if vector size >= 0 */
    void setVertices(const std::vector<Buffer::SP> &vertices,
                     const std::vector<Buffer::SP> &radii,
                     /*! the number of vertices in each time step */
                     size_t count);

    /*! pretty-print */
    std::string toString() const override;

    int vertexCount = 0;
    std::vector<Buffer::SP> verticesBuffers;
    std::vector<Buffer::SP> radiusBuffers;
  };

  // ------------------------------------------------------------------
  // implementation section
  // ------------------------------------------------------------------
  
  /*! get reference to given device-specific data for this object */
  inline SphereGeomType::DeviceData &
  SphereGeomType::getDD(const DeviceContext::SP &device) const
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<SphereGeomType::DeviceData>();
  }

  /*! get reference to given device-specific data for this object */
  inline SphereGeom::DeviceData &
  SphereGeom::getDD(const DeviceContext::SP &device)
  {
    assert(device && device->ID >= 0 && device->ID < (int)deviceData.size());
    return deviceData[device->ID]->as<SphereGeom::DeviceData>();
  }

  /*! get reference to the device-specific data for this object's *type* descriptor */
  inline SphereGeomType::DeviceData &
  SphereGeom::getTypeDD(const DeviceContext::SP &device) const
  {
    return (SphereGeomType::DeviceData &)type->getDD(device);
  }
  
} // ::owl
