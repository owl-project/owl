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

#if defined(_MSC_VER)
#  define GDT_VIEWER_DLL_EXPORT __declspec(dllexport)
#  define GDT_VIEWER_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define GDT_VIEWER_DLL_EXPORT __attribute__((visibility("default")))
#  define GDT_VIEWER_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define GDT_VIEWER_DLL_EXPORT
#  define GDT_VIEWER_DLL_IMPORT
#endif

#if defined(gdt_viewer_DLL_INTERFACE)
#  ifdef gdt_viewer_EXPORTS
#    define GDT_VIEWER_INTERFACE GDT_VIEWER_DLL_EXPORT
#  else
#    define GDT_VIEWER_INTERFACE GDT_VIEWER_DLL_IMPORT
#  endif
#else
#  define GDT_VIEWER_INTERFACE /*static lib*/
#endif
#include "gdt/math/box.h"
#include "gdt/math/LinearSpace.h"

