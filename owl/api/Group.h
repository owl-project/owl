// ======================================================================== //
// Copyright 2019 Ingo Wald                                                 //
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
#include "Geometry.h"

namespace owl {

  struct Group : public RegisteredObject {
    typedef std::shared_ptr<Group> SP;
    
    Group(Context *const context,
          ObjectRegistry &registry)
      : RegisteredObject(context,registry)
    {}
    virtual std::string toString() const { return "Group"; }
    virtual void buildAccel();

    OptixTraversableHandle getTraversable(int deviceID);
  };

  
  struct GeomGroup : public Group {
    typedef std::shared_ptr<GeomGroup> SP;

    GeomGroup(Context *const context,
              size_t numChildren);
    void setChild(int childID, Geom::SP child);
    
    virtual std::string toString() const { return "GeomGroup"; }
    std::vector<Geom::SP> geometries;
  };

  struct TrianglesGeomGroup : public GeomGroup {
    TrianglesGeomGroup(Context *const context,
                   size_t numChildren);
    virtual std::string toString() const { return "TrianglesGeomGroup"; }
  };

  struct UserGeomGroup : public GeomGroup {
    UserGeomGroup(Context *const context,
                   size_t numChildren);
    virtual std::string toString() const { return "UserGeomGroup"; }
    virtual void buildAccel() override;
  };

  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;
    
    InstanceGroup(Context *const context,
                  size_t numChildren,
                  Group::SP      *groups,
                  const uint32_t *instIDs,
                  const float    *xfms,
                  OWLMatrixFormat matrixFormat);

    void setChild(int childID, Group::SP child);
                  
    /*! set transformation matrix of given child */
    void setTransform(int childID, const affine3f &xfm);
    
    virtual std::string toString() const { return "InstanceGroup"; }

    /*! the list of children - note we do have to keep them both in
        the ll layer _and_ here for the refcounting to work; the
        transforms are only stored once, on the ll layer */
    std::vector<Group::SP> children;
  };

} // ::owl
