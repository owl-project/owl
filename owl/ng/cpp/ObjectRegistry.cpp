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

#include "ObjectRegistry.h"
#include "RegisteredObject.h"
#include "Context.h"

#include "Buffer.h"
#include "Group.h"
#include "RayGen.h"
#include "MissProg.h"

namespace owl {

  void ObjectRegistry::forget(RegisteredObject *object)
  {
    assert(object);
      
    std::lock_guard<std::mutex> lock(mutex);
    assert(object->ID >= 0);
    assert(object->ID < objects.size());
    assert(objects[object->ID] == object);
    objects[object->ID] = nullptr;
      
    previouslyReleasedIDs.push(object->ID);
  }
    
  void ObjectRegistry::track(RegisteredObject *object)
  {
    assert(object);
    std::lock_guard<std::mutex> lock(mutex);
    assert(object->ID >= 0);
    assert(object->ID < objects.size());
    assert(objects[object->ID] == nullptr);
    objects[object->ID] = object;
  }
    
  int ObjectRegistry::allocID()
  {
    std::lock_guard<std::mutex> lock(mutex);
    if (previouslyReleasedIDs.empty()) {
      objects.push_back(nullptr);
      const int newID = objects.size()-1;
      if (newID >= numIDsAllocedInContext) {
        while (newID >= numIDsAllocedInContext)
          numIDsAllocedInContext = std::max(1,numIDsAllocedInContext*2);
        std::cout << "re-allocing context IDs : " << numIDsAllocedInContext << std::endl;
        reallocContextIDs(numIDsAllocedInContext);
      }
      return newID;
    } else {
      int reusedID = previouslyReleasedIDs.top();
      previouslyReleasedIDs.pop();
      return reusedID;
    }
  }
  
  RegisteredObject *ObjectRegistry::getPtr(int ID)
  {
    std::lock_guard<std::mutex> lock(mutex);
      
    assert(ID >= 0);
    assert(ID < objects.size());
    assert(objects[ID]);
    return objects[ID];
  }

  template<>
  void ObjectRegistryT<Buffer>::reallocContextIDs(int newMaxIDs)
  {
    std::cout << "#ng: re-allocing buffer IDs to " << newMaxIDs << std::endl;
    lloAllocBuffers(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<Module>::reallocContextIDs(int newMaxIDs)
  {
    lloAllocModules(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<Geom>::reallocContextIDs(int newMaxIDs)
  {
    std::cout << "#ng: re-allocing geoms to " << newMaxIDs << std::endl;
    lloAllocGeoms(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<GeomType>::reallocContextIDs(int newMaxIDs)
  {
    lloAllocGeomTypes(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<RayGen>::reallocContextIDs(int newMaxIDs)
  {
    std::cout << "#ng: re-allocing rayGens to " << newMaxIDs << std::endl;
    lloAllocRayGens(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<RayGenType>::reallocContextIDs(int newMaxIDs)
  {
    /* ray gen types are not tracked by ll context, so need no alloc */
  }

  template<>
  void ObjectRegistryT<MissProg>::reallocContextIDs(int newMaxIDs)
  {
    std::cout << "#ng: re-allocing missProgs to " << newMaxIDs << std::endl;
    lloAllocMissProgs(context->llo,newMaxIDs);
  }

  template<>
  void ObjectRegistryT<MissProgType>::reallocContextIDs(int newMaxIDs)
  {
    OWL_NOTIMPLEMENTED;
  }

  template<>
  void ObjectRegistryT<Group>::reallocContextIDs(int newMaxIDs)
  {
    std::cout << "#ng: re-allocing groups to " << newMaxIDs << std::endl;
    lloAllocGroups(context->llo,newMaxIDs);
  }

} // ::owl

