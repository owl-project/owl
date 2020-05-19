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

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif()
project(Optix-OWL VERSION 0.7.4)

#OPTION(OWL_BUILD_SAMPLES "Build the Samples?" OFF)
OPTION(OWL_BUILD_ADVANCED_TESTS "Build the *advanced* test-cases?" OFF)
OPTION(OWL_USE_TBB "Use TBB to parallelize computation?" ON)

cmake_minimum_required(VERSION 2.8)
# iw, 1/3/20: yes, i know this is 'old' cmake style, but on CentOS7
# this is the only way that works, since the 'correct' way of setting
# the cxx11 standard doesnt propagate those flags to nvcc for the cuda
# programs. TODO: check to see if this can be fixed in the
# configure_cuda script instead.

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT WIN32)
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(owl_dir ${PROJECT_SOURCE_DIR})

include_directories(${PROJECT_SOURCE_DIR}/owl/include)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${owl_dir}/owl/common/cmake/")

include(configure_cuda)
include(configure_optix)
include(configure_owl)
include(configure_build_type)

include_directories(${OWL_INCLUDES})

# ------------------------------------------------------------------
# owl library itself, including owl-common
# ------------------------------------------------------------------
add_subdirectory(owl)

# ------------------------------------------------------------------
# tutorial/samples
# ------------------------------------------------------------------
#if (OWL_BUILD_SAMPLES)
  enable_testing()
  add_subdirectory(samples)
#endif()

# ------------------------------------------------------------------
# some basic testing
# ------------------------------------------------------------------
if (OWL_BUILD_ADVANCED_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()

# ------------------------------------------------------------------
# check if this is included as a submodule, and if so, export some
# variables to the parent
# ------------------------------------------------------------------
if (NOT (${PROJECT_SOURCE_DIR} STREQUAL ${CMAKE_HOME_DIRECTORY}))
  message("#owl.cmake: exporting OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES}")
  set(OWL_INCLUDES ${OWL_INCLUDES} PARENT_SCOPE)
  set(OWL_VIEWER_INCLUDES ${OWL_VIEWER_INCLUDES} PARENT_SCOPE)
  set(OWL_LIBRARIES ${OWL_LIBRARIES} PARENT_SCOPE)
  set(OWL_VIEWER_LIBRARIES ${OWL_VIEWER_LIBRARIES} PARENT_SCOPE)
endif()
