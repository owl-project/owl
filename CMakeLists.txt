# ======================================================================== #
# Copyright 2019-2020 Ingo Wald                                            #
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

# If using this option, set OWL_CUDA_ARCH to a single value like 61 or 86.
option(OWL_USE_INTERMEDIATE_CMAKE "(3.8 <= CMake <= 3.17) Use first-class CUDA CMake build but before CMAKE_CUDA_ARCHITECTURES was introduced" OFF)
# If using this option, set CMAKE_CUDA_ARCHITECTURES to a single value like 61 or 86.
option(OWL_USE_MODERN_CMAKE "(CMake >= 3.18) Use modern first-class CUDA CMake build" OFF)

OPTION(OWL_BUILD_SAMPLES "Build the Samples?" ON)
OPTION(OWL_BUILD_ADVANCED_TESTS "Build the *advanced* test-cases?" OFF)
OPTION(OWL_USE_TBB "Use TBB to parallelize computation?" ON)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_HOME_DIRECTORY}))
  SET(OWL_IS_SUBPROJECT ON)
else()
  SET(OWL_IS_SUBPROJECT OFF)
endif()

if(CUDA_TOOLKIT_ROOT_DIR)
  # Somebody called find_cuda--they are trying to use the deprecated build.
  message(#SEND_ERROR
    WARNING
    "CUDA_TOOLKIT_ROOT_DIR is set. "
    "This usually means that a parent project has called find_cuda, which "
    "is deprecated and not supported by this 'modern cmake' version of OWL. "
    "Instead, the parent project should use modern cmake methodology, too,"
    "with CUDA first-class language support (i.e., project(foo LANGUAGES CXX CUDA))"
    )
endif()

if(OWL_USE_INTERMEDIATE_CMAKE)
  set(OWL_INTERMEDIATE_CMAKE ON)
elseif(OWL_USE_MODERN_CMAKE)
  set(OWL_MODERN_CMAKE ON)
else()
  # If none of these options are set, we need to guess which build you want.
  if(${CMAKE_VERSION} VERSION_LESS 3.18)
    set(OWL_INTERMEDIATE_CMAKE ON)
  else()
    message(STATUS "Defaulting to modern CMake (CMAKE_CUDA_ARCHITECTURES required) "
      "as you are running CMake >= 3.18. If this is not desired, set "
      "OWL_USE_INTERMEDIATE_CMAKE.")
    set(OWL_MODERN_CMAKE ON)
  endif()
endif()

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW) # Setting version in project command
endif()

if(OWL_INTERMEDIATE_CMAKE)
  cmake_minimum_required(VERSION 3.8)
  if(NOT ${CMAKE_VERSION} VERSION_LESS 3.18)
    cmake_policy(SET CMP0104 OLD) # CUDA arch set manually by compiler flags
  endif()
else()
  cmake_minimum_required(VERSION 3.18)
  cmake_policy(SET CMP0104 NEW) # CUDA_ARCHITECTURES must be specified
endif()
# project command is required to come after cmake_minimum_required command.
project(Optix-OWL VERSION 1.0.6 LANGUAGES C CXX CUDA) # c language required for embed_ptx

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(owl_dir ${PROJECT_SOURCE_DIR})
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/cmake/")

# if not a subproject, set "Debug/Release flags and binary output dir
# to <binaryDir>
include(configure_build_type)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(OWL_CXX_FLAGS "")

if (WIN32)
  # Nate: 11/3/20: I get "fatal error C1041: cannot open program database" 
  # on windows from multiple targets writing to the same .pdb file simultaneously.
  # This flag doesn't seem to fix it.
  # set (CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} "/FS")
endif()

#set(CMAKE_POSITION_INDEPENDENT_CODE ON) # moved to configure_cuda.cmake

include(configure_cuda)
include(configure_optix) # This must follow configure_cuda
include(configure_owl) # This must follow configure_optix
include(configure_tbb) # This must follow configure_owl

include_directories(${OWL_INCLUDES})
add_definitions(${OWL_CXX_FLAGS})

# ------------------------------------------------------------------
# owl library itself, including owl-common
# ------------------------------------------------------------------
add_subdirectory(owl)

# ------------------------------------------------------------------
# tutorial/samples
# ------------------------------------------------------------------
if (OWL_BUILD_SAMPLES)
  enable_testing()
  add_subdirectory(samples)
endif()

# ------------------------------------------------------------------
# some basic testing
# ------------------------------------------------------------------
if (NOT OWL_IS_SUBPROJECT)
  if (OWL_BUILD_ADVANCED_TESTS)
    enable_testing()
    add_subdirectory(tests)
  endif()
endif()

# ------------------------------------------------------------------
# check if this is included as a submodule, and if so, export some
# variables to the parent
# ------------------------------------------------------------------
if (OWL_IS_SUBPROJECT)
  set(OWL_INCLUDES ${OWL_INCLUDES} PARENT_SCOPE)
  set(OWL_VIEWER_INCLUDES ${OWL_VIEWER_INCLUDES} PARENT_SCOPE)
  set(OWL_LIBRARIES ${OWL_LIBRARIES} PARENT_SCOPE)
  set(OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES} PARENT_SCOPE)
  set(OWL_HAVE_TBB ${OWL_HAVE_TBB} PARENT_SCOPE)
  set(OWL_CXX_FLAGS ${OWL_CXX_FLAGS} PARENT_SCOPE)
endif()
