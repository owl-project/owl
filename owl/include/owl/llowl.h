// ======================================================================== //
// Copyright 2019-2020 Ingo Wald                                            //
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

/*! \file include/owl/ll.h Defines the dynamically linkable "C-API"
 *  for the low-level owl::ll abstraction layer - most users should
 *  use owl.h / owl_host.h instead */

#pragma once

#ifndef _USE_MATH_DEFINES
#  define _USE_MATH_DEFINES
#endif
#include <math.h> // using cmath causes issues under Windows
#ifdef _WIN32
#else
#  include <unistd.h>
#endif

#include <iostream>
#include <stdint.h>

#ifdef _WIN32
#pragma warning( push )
#pragma warning( disable : 4996 )
#endif

#include <cuda.h>
#include <driver_types.h>
#include <optix.h>
#ifdef _WIN32
#pragma warning( push )
#endif

#if defined(_MSC_VER)
#  define OWL_LL_DLL_EXPORT __declspec(dllexport)
#  define OWL_LL_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_LL_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_LL_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_LL_DLL_EXPORT
#  define OWL_LL_DLL_IMPORT
#endif

#ifdef llowl_EXPORTS
#  define OWL_LL_INTERFACE OWL_LL_DLL_EXPORT
#else
#  define OWL_LL_INTERFACE OWL_LL_DLL_IMPORT
#endif


