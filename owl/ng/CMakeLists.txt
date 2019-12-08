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

include_directories(${PROJECT_SOURCE_DIR}/owl)
include_directories(${PROJECT_SOURCE_DIR}/owl/ng)
include_directories(${PROJECT_SOURCE_DIR}/owl/ng/include)

add_library(owl-ng

  include/owl/owl.h
  
  api/impl.cpp
  api/APIContext.h
  api/APIContext.cpp
  api/APIHandle.h
  api/APIHandle.cpp
  
  cpp/Object.cpp
  cpp/RegisteredObject.cpp
  cpp/SBTObject.cpp
  cpp/Buffer.cpp
  cpp/Group.cpp
  cpp/ObjectRegistry.cpp
  cpp/Context.cpp
  cpp/RayGen.cpp
  cpp/Geometry.cpp
  cpp/Variable.cpp
  )

include_directories(${PROJECT_SOURCE_DIR}/owl/include)

target_link_libraries(owl-ng
  owl-ll
  )





