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

set(OWL_NG_SOURCES

  ../include/owl/owl.h
  
  api/impl.cpp
  api/APIContext.h
  api/APIContext.cpp
  api/APIHandle.h
  api/APIHandle.cpp
  
  cpp/Object.cpp
  cpp/Module.cpp
  cpp/RegisteredObject.cpp
  cpp/SBTObject.cpp
  cpp/Buffer.cpp
  cpp/Group.cpp
  cpp/ObjectRegistry.cpp
  cpp/Context.cpp
  cpp/RayGen.cpp
  cpp/LaunchParams.cpp
  cpp/MissProg.cpp
  cpp/Geometry.cpp
  cpp/Variable.cpp
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





