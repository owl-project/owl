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

include(configure_owl)

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

cuda_add_library(toneMap STATIC
  toneMap.cu)
add_executable(adv_optix7course
  ${embedded_ptx_code}
  devicePrograms.cu
  optix7.h
  CUDABuffer.h
  SampleRenderer.h
  SampleRenderer.cpp
  Model.cpp
  main.cpp
  )
#add_subdirectory(glfWindow)

find_package(OpenGL)
message("optix 7 course : OPENGL_gl_LIBRARY =  ${OPENGL_gl_LIBRARY}")
message("optix 7 course : OWL_VIEWER_LIBRARIES = ${OWL_VIEWER_LIBRARIES}")

target_link_libraries(adv_optix7course
  toneMap
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # owl viewer, and everything it requires
  ${OWL_VIEWER_LIBRARIES}
  )
