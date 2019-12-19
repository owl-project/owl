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

set(OWL_LL_SOURCES
  common.h

  Device.h
  Device.cpp

  TrianglesGeomGroup.cpp
  UserGeomGroup.cpp
  InstanceGroup.cpp
  
  DeviceGroup.h
  DeviceGroup.cpp
  )

add_library(owl-ll-static STATIC
  ${OWL_LL_SOURCES}
  )

add_library(owl-ll SHARED
  ${OWL_LL_SOURCES}
  )
target_link_libraries(owl-ll
#  owl-common
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  )
set_target_properties(owl-ll PROPERTIES VERSION ${PROJECT_VERSION})
set_target_properties(owl-ll PROPERTIES CXX_VISIBILITY_PRESET hidden)
target_compile_definitions(owl-ll PUBLIC PBRT_PARSER_DLL_INTERFACE)






