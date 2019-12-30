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
#  define OWL_VIEWER_DLL_EXPORT __declspec(dllexport)
#  define OWL_VIEWER_DLL_IMPORT __declspec(dllimport)
#elif defined(__clang__) || defined(__GNUC__)
#  define OWL_VIEWER_DLL_EXPORT __attribute__((visibility("default")))
#  define OWL_VIEWER_DLL_IMPORT __attribute__((visibility("default")))
#else
#  define OWL_VIEWER_DLL_EXPORT
#  define OWL_VIEWER_DLL_IMPORT
#endif

#if defined(owl_viewer_DLL_INTERFACE)
#  ifdef owl_viewer_EXPORTS
#    define OWL_VIEWER_INTERFACE OWL_VIEWER_DLL_EXPORT
#  else
#    define OWL_VIEWER_INTERFACE OWL_VIEWER_DLL_IMPORT
#  endif
#else
#  define OWL_VIEWER_INTERFACE /*static lib*/
#endif
#include "owl/common/math/box.h"
#include "owl/common/math/LinearSpace.h"

