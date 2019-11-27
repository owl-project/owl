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

#include "Object.h"

namespace owl {

  struct RegisteredObject;
  
  /*! registry that tracks mapping between buffers and buffer
    IDs. Every buffer should have a valid ID, and should be tracked
    in this registry under this ID */
  struct ObjectRegistry {
    inline size_t size() const { return objects.size(); }
    inline bool   empty() const { return objects.empty(); }
    
    void forget(RegisteredObject *object);
    void track(RegisteredObject *object);
    int allocID();
    RegisteredObject *getPtr(int ID);

  private:
    /*! list of all tracked objects. note this are *NOT* shared-ptr's,
      else we'd never released objects because each object would
      always be owned by the registry */
    std::vector<RegisteredObject *> objects;
    
    /*! list of IDs that have already been allocated before, and have
      since gotten freed, so can be re-used */
    std::stack<int> previouslyReleasedIDs;
    std::mutex mutex;
  };


  /*! registry that tracks mapping between buffers and buffer
    IDs. Every buffer should have a valid ID, and should be tracked
    in this registry under this ID */
  template<typename T>
  struct ObjectRegistryT : public ObjectRegistry {
    inline T *getPtr(int ID)
    { return (T*)ObjectRegistry::getPtr(ID); }

    inline typename T::SP getSP(int ID)
    {
      T *ptr = getPtr(ID);
      assert(ptr);
      Object::SP object = ptr->shared_from_this();
      assert(object);
      return object->as<T>();
      // return ptr->shared_from_this()->as<T>();
    }
  };
    
} // ::owl

