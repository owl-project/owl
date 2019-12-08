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
      return objects.size()-1;
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
    
} // ::owl

