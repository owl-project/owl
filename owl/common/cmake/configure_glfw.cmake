# ======================================================================== #
# Copyright 2020 Ingo Wald                                                 #
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

# configures the glfw library

set(OWL_HAVE_GLFW OFF)

message("BEFORE configure_glfw : OpenGL_gl_LIBRARY =  ${OpenGL_gl_LIBRARY}")
message("BEFORE configure_glfw : OPENGL_LIBRARIES =  ${OPENGL_LIBRARIES}")

set(OpenGL_GL_PREFERENCE "GLVND")
set(OpenGL_GL_PREFERENCE "GLVND" PARENT_SCOPE)
find_package(OpenGL)

message("configure_glfw : OpenGL_gl_LIBRARY =  ${OpenGL_gl_LIBRARY}")
message("configure_glfw : OPENGL_LIBRARIES =  ${OPENGL_LIBRARIES}")


if (OpenGL_FOUND)
  find_package(glfw3)
  
  if (${glfw3_FOUND})
    message("Found GLFW3 Package")
    include_directories(${glfw3_DIR})
    set(OWL_HAVE_GLFW ON)
    set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
    set(OWL_GLFW_INCLUDES ${glfw3_INCLUDES})

  else()
    message("Found OpenGL, but did NOT find glfw3 in system - building glfw from source")
    add_subdirectory(${owl_dir}/samples/common/3rdParty/glfw)
    set(OWL_GLFW_INCLUDES ${owl_dir}/samples/common/3rdParty/glfw)
    set(OWL_GLFW_LIBRARIES glfw ${OPENGL_LIBRARIES})
  endif()
endif()

message("configure_glfw: OWL_HAVE_GLFW ${OWL_HAVE_GLFW} ")
set(OWL_HAVE_GLFW ${OWL_HAVE_GLFW} PARENT_SCOPE)

