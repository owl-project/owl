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

include_directories(${OWL_VIEWER_INCLUDES})

cuda_compile_and_embed(embedded_ptx_code devicePrograms.cu)

add_executable(adv_optix7course
  # the file with embedded PTX string for the device programs:
  ${embedded_ptx_code}
  devicePrograms.cu # <-- note this doesn't actually get compiled in, only as string of PTX instrucions (see comile_and_embed above); this is only here for Visual Studio GUI to show it as an input

  SampleRenderer.h
  SampleRenderer.cpp
  Model.h
  Model.cpp
  LaunchParams.h
  main.cpp
  )

target_link_libraries(adv_optix7course
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # owl and owl viewer, and everything these require
  ${OWL_LIBRARIES}
  ${OWL_VIEWER_LIBRARIES}
  )
