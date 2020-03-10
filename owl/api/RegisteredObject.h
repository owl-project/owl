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

  struct ObjectRegistry;

  /*! a object that is managed/kept track of in a registry that
      assigns linear IDs (so that, for example, the SBT builder can
      easily iterate over all geometries, all geometry types, etc. The
      sole job of this class is to properly register and unregister
      itself in the given registry when it gets created/destroyed */
  struct RegisteredObject : public ContextObject {
    RegisteredObject(Context *const context,
                     ObjectRegistry &registry);
    ~RegisteredObject();

    int             ID;
    ObjectRegistry &registry;
  };

} // ::owl

