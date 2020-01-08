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

include_directories(${PROJECT_SOURCE_DIR}/)
# for STB image read/write
include_directories(${PROJECT_SOURCE_DIR}/owl/common/3rdParty)
# public API:
include_directories(${PROJECT_SOURCE_DIR}/owl/include/)

#option(BUILD_VISUAL_SAMPLES "Build Visual Examples (requires glut)" OFF)
#if(BUILD_VISUAL_SAMPLES)
#  add_subdirectory(../owl/common/viewerWidget sub_viewerWidget)
#endif()


# low-level interface
add_subdirectory(ll)

# node graph layer
add_subdirectory(ng)


option(BUILD_EX_OPTIX7COURSE "Build Siggraph 2018 Optix Course Version" OFF)
if (BUILD_EX_OPTIX7COURSE)
  if(BUILD_VISUAL_SAMPLES)
    add_subdirectory(advanced/optix7course)
  else ()
    message("optix7course example requires BUILD_VISUAL_EXAMPLES to be turned on")
  endif()
endif()

