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

#include "Object.h"

namespace owl {

  std::atomic<uint64_t> Object::nextAvailableID;

  Object::Object()
    : uniqueID(nextAvailableID++)
  {}
    
  std::string typeToString(OWLDataType type)
  {
    switch(type) {
    case OWL_INT:
      return "int";
    case OWL_FLOAT:
      return "float";
    case OWL_FLOAT3:
      return "float3";
    case OWL_BUFFER:
      return "OWLBuffer";
    case OWL_GROUP:
      return "OWLGroup";
    default:
      throw std::runtime_error(std::string(__PRETTY_FUNCTION__)
                               +": not yet implemented for type #"
                               +std::to_string((int)type));
    }
  }
  
} // ::owl


