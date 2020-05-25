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
    if (object->ID == -1)
      // object is already de-registered (eg, with an explicit
      // bufferdestroy, even if owl::buffer object still has a
      // reference count and thus hasn't been deleted yet.
      return;
    
    std::lock_guard<std::mutex> lock(mutex);
    assert(object->ID >= 0);
    assert(object->ID < objects.size());
    assert(objects[object->ID] == object);
    objects[object->ID] = nullptr;
      
    previouslyReleasedIDs.push(object->ID);

    object->ID = -1;
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
      const int newID = int(objects.size()-1);
      if (newID >= numIDsAllocedInContext) {
        while (newID >= numIDsAllocedInContext)
          numIDsAllocedInContext = std::max(1,numIDsAllocedInContext*2);
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
    context->llo->allocBuffers(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<Texture>::reallocContextIDs(int newMaxIDs)
  {
    /* nothing to do - textures not tracked by llo level */
  }

  template<>
  void ObjectRegistryT<Module>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocModules(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<Geom>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocGeoms(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<GeomType>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocGeomTypes(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<RayGen>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocRayGens(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<RayGenType>::reallocContextIDs(int newMaxIDs)
  {
    /* ray gen types are not tracked by ll context, so need no alloc */
  }

  template<>
  void ObjectRegistryT<MissProg>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocMissProgs(newMaxIDs);
  }

  template<>
  void ObjectRegistryT<MissProgType>::reallocContextIDs(int newMaxIDs)
  {
    /* nothign do to here!? */
  }

  template<>
  void ObjectRegistryT<LaunchParams>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocLaunchParams(newMaxIDs);
  }
  
  template<>
  void ObjectRegistryT<LaunchParamsType>::reallocContextIDs(int newMaxIDs)
  {
    /* nothign do to here!? */
  }

  template<>
  void ObjectRegistryT<Group>::reallocContextIDs(int newMaxIDs)
  {
    context->llo->allocGroups(newMaxIDs);
  }

} // ::owl

