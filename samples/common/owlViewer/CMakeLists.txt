# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
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

include(configure_glfw)

if (NOT OWL_HAVE_GLFW)
  return()
endif()

add_library(owl_viewer STATIC)

target_sources(owl_viewer PRIVATE
  # add header files, so visual studio will properly show them as part of the solution
  OWLViewer.h
  Camera.h
  InspectMode.h
  FlyMode.h

  # the actual source files
  OWLViewer.cpp
  Camera.cpp
  InspectMode.cpp
  FlyMode.cpp
)

target_link_libraries(owl_viewer
  PUBLIC
    owl::owl
    ${OWL_GLFW_LIBRARIES}
)

target_include_directories(owl_viewer
  PUBLIC
    ${OWL_GLFW_INCLUDES}
    ${owl_dir}/samples/common
)
