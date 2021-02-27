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

if(OWL_DEPRECATED_CMAKE)
  cuda_compile_and_embed(embedded_ptx_code
    devicePrograms.cu
    )
else()
  embed_ptx(embedded_ptx_code devicePrograms.cu)
endif()

if(OWL_DEPRECATED_CMAKE)
  cuda_add_library(toneMap STATIC
	toneMap.cu)
else()
  add_library(toneMap STATIC toneMap.cu)
endif()

set(sourcefiles
  ${embedded_ptx_code}
  optix7.h
  CUDABuffer.h
  SampleRenderer.h
  SampleRenderer.cpp
  Model.cpp
  main.cpp
  )
  
# lp: 2/25/21: devicePrograms.cu should not be compiled and linked as a 
# normal CUDA binary, the PTX intrinsics for e.g. optixGetSbtDataPointer
# are not defined in normal CUDA kernels!
if(OWL_DEPRECATED_CMAKE)
  set(sourcefiles "${sourcefiles}" devicePrograms.cu)
endif()

add_executable(adv_optix7course
  ${sourcefiles}
  )

#target_compile_options(adv_optix7course PRIVATE "--keep")

if(OWL_DEPRECATED_CMAKE)
  set(extra_cuda_libs ${CUDA_LIBRARIES} ${CUDA_CUDA_LIBRARY})
endif()
target_link_libraries(adv_optix7course
  toneMap
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${extra_cuda_libs}
  # owl viewer, and everything it requires
  ${OWL_VIEWER_LIBRARIES})
