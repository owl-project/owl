// ======================================================================== //
// Copyright 2018 Ingo Wald                                                 //
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

#include "gdt/cuda.h"
#include "gdt/gdt.h"

#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ 1
#include <optix.h>
#define __CUDA_INCLUDE_COMPILER_INTERNAL_HEADERS__ 1

#ifndef GDT_USE_OPTIX_7
#include <optixu/optixpp.h>
#else
#include <optix_host.h>
#endif
