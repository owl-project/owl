# ======================================================================== #
# Copyright 2019-2021 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

cmake_minimum_required(VERSION 3.9)

cmake_policy(SET CMP0048 NEW) # Setting version in project command
#cmake_policy(SET CMP0104 OLD) # CUDA arch set manually by compiler flags
cmake_policy(SET CMP0104 NEW) # CUDA_ARCHITECTURES must be specified

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  SET(OWL_IS_SUBPROJECT ON)
else()
  SET(OWL_IS_SUBPROJECT OFF)
endif()

if(CUDA_TOOLKIT_ROOT_DIR)
  # Somebody called find_cuda--they are trying to use the deprecated build.
  message(WARNING "CUDA_TOOLKIT_ROOT_DIR is set. "
    "This usually means that a parent project has called find_cuda, which "
    "has been deprecated in CMake since 2018 and is no longer supported by OWL. "
    "Instead, the parent project is required to use 'intermediate' or 'modern' "
    "CMake CUDA style with first-class CUDA language support (i.e., "
    "project(foo LANGUAGES C CXX CUDA)). "
    "See OWL's CMakeLists.txt and owl/cmake/configure_cuda.cmake for more info "
    "about the differences between 'intermediate' and 'modern' style."
    )
endif()

# project command is required to come after cmake_minimum_required command.
project(Optix-OWL VERSION 1.1.0 LANGUAGES C CXX CUDA)

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/owl/cmake/")

# if not a subproject, set "Debug/Release flags and binary output dir
# to <binaryDir>
include(configure_build_type)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(embed_ptx)

# ------------------------------------------------------------------
# owl library itself, including owl-common
# ------------------------------------------------------------------
add_subdirectory(owl)

if (OWL_IS_SUBPROJECT)
  return()
endif()

option(OWL_BUILD_SAMPLES "Build the Samples?" ON)
option(OWL_BUILD_ADVANCED_TESTS "Build the *advanced* test-cases?" OFF)

add_subdirectory(3rdParty)

# ------------------------------------------------------------------
# tutorial/samples
# ------------------------------------------------------------------
if (OWL_BUILD_SAMPLES)
  add_subdirectory(samples)
endif()

# ------------------------------------------------------------------
# some basic testing
# ------------------------------------------------------------------
if (OWL_BUILD_ADVANCED_TESTS)
  add_subdirectory(tests)
endif()
