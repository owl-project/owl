# ======================================================================== #
# Copyright 2024-2024 Ingo Wald                                            #
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

function(pyowl_compile_ptx output_ptx input_cu)
  add_library(${output_ptx} OBJECT)
  target_sources(${output_ptx} PRIVATE ${COMPILE_PTX_SOURCES})
  set_property(TARGET ${output_ptx} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${output_ptx} PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_options(${output_ptx} PRIVATE "-lineinfo")
  target_sources(${output_ptx} PRIVATE ${input_cu})
  add_custom_target(build_${output_ptx} ALL DEPENDS ${output_ptx}
    COMMAND  cp "$<TARGET_OBJECTS:${output_ptx}>" ${PROJECT_BINARY_DIR}/${output_ptx}
    BYPRODUCTS ${PROJECT_BINARY_DIR}/${output_ptx}
    VERBATIM)
endfunction()

# ------------------------------------------------------------------
pyowl_compile_ptx(cmd01_simpleTriangles.ptx cmd01_simpleTriangles.cu)
target_link_libraries(cmd01_simpleTriangles.ptx PUBLIC owl::owl)

# ------------------------------------------------------------------
pyowl_compile_ptx(cmd05_rtow.ptx cmd05_rtow.cu)
target_link_libraries(cmd05_rtow.ptx PUBLIC owl::owl)

