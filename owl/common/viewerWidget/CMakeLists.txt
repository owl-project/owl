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

include(../cmake/configure_glut.cmake)

if(GLUT_FOUND)
  include_directories(${GLUT_INCLUDE_DIR})
endif()

set(owl_viewerWidget_sources
  # add header files, so visual studio will properly show them as part of the solution
  ViewerWidget.h
  Camera.h
  InspectMode.h
  FlyMode.h
  GlutWindow.h

  # the actual source files
  ViewerWidget.cpp
  Camera.cpp
  InspectMode.cpp
  FlyMode.cpp
  GlutWindow.cpp
)
# this is doing the same using OptiX
add_library(owl_viewerWidget SHARED ${owl_viewerWidget_sources})
add_library(owl_viewerWidget_static STATIC ${owl_viewerWidget_sources})
  
target_link_libraries(owl_viewerWidget
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  llowl
  )
target_compile_definitions(owl_viewerWidget PUBLIC owl_viewerWidget_DLL_INTERFACE)

target_link_libraries(owl_viewerWidget_static
  ${OPENGL_LIBRARIES}
  ${GLUT_LIBRARIES}
  llowl_static
  )
