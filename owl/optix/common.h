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

// gdt
#include "gdt/math/AffineSpace.h"
#include "gdt/parallel/parallel_for.h"
// std
#include <vector>
#include <memory>
#include <mutex>
#include <map>

#define OWL_NOTIMPLEMENTED throw std::runtime_error(std::string(__PRETTY_FUNCTION__)+\
                                                    " : not yet implemented")


