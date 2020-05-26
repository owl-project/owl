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

set(OWL_NG_SOURCES

  ../include/owl/owl.h
  ../common/cmake/configure_tbb.cmake
  ../common/cmake/configure_owl.cmake
  ../common/cmake/configure_optix.cmake
  ../common/cmake/FindOptiX.cmake

  impl.cpp
  APIContext.h
  APIContext.cpp
  APIHandle.h
  APIHandle.cpp
  
  Object.cpp
  Module.cpp
  RegisteredObject.cpp
  SBTObject.cpp
  Buffer.cpp
  Texture.cpp
  Group.cpp
  ObjectRegistry.cpp
  Context.cpp
  RayGen.cpp
  LaunchParams.cpp
  MissProg.cpp
  Geometry.cpp
  Variable.cpp
)

add_library(owl
${OWL_NG_SOURCES}
  )
add_library(owl_static STATIC
${OWL_NG_SOURCES}
  )

include_directories(${PROJECT_SOURCE_DIR}/owl/include)

target_link_libraries(owl
  llowl_static
  )

target_link_libraries(owl_static
  llowl_static
  )


set_target_properties(owl PROPERTIES POSITION_INDEPENDENT_CODE ON)
set_target_properties(owl_static PROPERTIES POSITION_INDEPENDENT_CODE ON)





