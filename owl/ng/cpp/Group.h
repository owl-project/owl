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
    virtual void buildAccel() { IGNORING_THIS(); }
  };

  
  struct GeomGroup : public Group {
    typedef std::shared_ptr<GeomGroup> SP;

    GeomGroup(Context *const context,
              size_t numChildren);
    void setChild(int childID, Geom::SP child)
    {
      assert(childID >= 0);
      assert(childID < geometries.size());
      geometries[childID] = child;
    }
    virtual std::string toString() const { return "GeomGroup"; }
    std::vector<Geom::SP> geometries;
  };

  struct TrianglesGroup : public GeomGroup {
    TrianglesGroup(Context *const context,
                   size_t numChildren);
    virtual std::string toString() const { return "TrianglesGroup"; }
  };

  struct InstanceGroup : public Group {
    typedef std::shared_ptr<InstanceGroup> SP;
    
    InstanceGroup(Context *const context,
                  size_t numChildren);
    void setChild(int childID, Group::SP child)
    {
      assert(childID >= 0);
      assert(childID < children.size());
      children[childID] = child;
    }
    virtual std::string toString() const { return "InstanceGroup"; }
    std::vector<Group::SP> children;
  };

} // ::owl
