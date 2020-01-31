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

include_directories(${PROJECT_SOURCE_DIR}/)
# for STB image read/write
include_directories(${PROJECT_SOURCE_DIR}/owl/common/3rdParty)
# public API:
include_directories(${PROJECT_SOURCE_DIR}/owl/include/)

# ---------------------------------------------------------------------------
# the default "tutorial-style" samples that intoduce, step by step,
# individual capabililies like creating a raygen (s00), creating a
# triangle mesh (s01), ... instancing (s08), etc.
# ---------------------------------------------------------------------------
file(GLOB samples RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} "s??-*")

foreach(sample ${samples})
  add_subdirectory(${sample})
endforeach()

# ---------------------------------------------------------------------------
# a more serious kind of example with model loader, viewer, simple
# path tracer, etc
# ---------------------------------------------------------------------------
option(BUILD_EX_OPTIX7COURSE "Build Siggraph 2018 Optix Course Version" OFF)
if (BUILD_EX_OPTIX7COURSE)
  add_subdirectory(advanced/optix7course)
endif()


