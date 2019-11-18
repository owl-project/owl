# ======================================================================== #
# Copyright 2019 Ingo Wald                                                 #
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

cmake_policy(SET CMP0048 NEW)
project(Optix VERSION 0.0.1)

cmake_minimum_required(VERSION 2.8)

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/submodules/gdt/)

# -------------------------------------------------------
set(gdt_dir ${CMAKE_SOURCE_DIR}/submodules/gdt)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")

include(configure_cuda)
include(configure_optix)
#include(configure_embree)
include(configure_tbb)
include(configure_build_type)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------------------------------------------
# import gdt submodules itself: gdt, and viewer
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# -------------------------------------------------------
# optix itself
# -------------------------------------------------------
#configure_file(optix/config.h.in ${CMAKE_BINARY_DIR}/optix/optix_config.h)

include_directories(${CMAKE_BINARY_DIR})
#include_directories(${PROJECT_SOURCE_DIR}/owl/)
#include_directories(${PROJECT_SOURCE_DIR}/optix/include/)

# the actual back-end
add_subdirectory(owl/optix)

# the front-end apps: viewer, tools, etc
add_subdirectory(samples)
