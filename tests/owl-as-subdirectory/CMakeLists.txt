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

cmake_minimum_required(VERSION 3.9)

cmake_policy(SET CMP0104 NEW) # CUDA_ARCHITECTURES must be specified

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(owlSubmoduleTest LANGUAGES C CXX CUDA)

# This is the same as someone calling add_subdirectory(path/to/OWL) submodule
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/../..
  ${CMAKE_BINARY_DIR}/owl
)

# Parent projects must include embed_ptx.cmake in order to use it
include(embed_ptx)

# NOTE(jda) - Hack here to get stb_image...would be provided by parent project
#             normally.
add_subdirectory(
  ${CMAKE_SOURCE_DIR}/../../3rdParty/stb_image
  ${CMAKE_BINARY_DIR}/stb_image
)

# Build the many-spheres test to verify things work
add_subdirectory(
  ${CMAKE_CURRENT_LIST_DIR}/../t01-many-spheres
  ${CMAKE_BINARY_DIR}/t01-many-spheres
)
