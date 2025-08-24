# ======================================================================== #
# Copyright 2019-2025 Ingo Wald                                            #
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


# ------------------------------------------------------------------
# include owl-based implementation of owl_prime API
# ------------------------------------------------------------------
add_library(owl_prime-interface INTERFACE)
target_include_directories(owl_prime-interface INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
 )
target_link_libraries(owl_prime-interface INTERFACE owl::owl)

# ------------------------------------------------------------------
# build abstract owl_prime abstraction layer that implements the API.
# ------------------------------------------------------------------
set(SOURCES
  primer-common/Geom.h
  primer-common/Group.h
  primer-common/Context.h
  primer-common/primer.cpp
  )

add_library(owl_prime_static STATIC ${SOURCES})
target_link_libraries(owl_prime_static PUBLIC owl_prime-interface)
set_target_properties(owl_prime_static
  PROPERTIES
  CUDA_USE_STATIC_RUNTIME ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )
target_link_libraries(owl_prime_static PUBLIC CUDA::cuda_driver CUDA::cudart_static)

add_library(owl_prime SHARED ${SOURCES})
target_link_libraries(owl_prime PUBLIC owl_prime-interface)
set_target_properties(owl_prime
  PROPERTIES
  CUDA_USE_STATIC_RUNTIME ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
#  CUDA_ARCHITECTURES OFF
  )
find_package(CUDAToolkit REQUIRED)
target_link_libraries(owl_prime PUBLIC CUDA::cuda_driver CUDA::cudart_static)

# allow actual backend(s) in subdirector(ies) to link to this library
# in current directory
cmake_policy(SET CMP0079 NEW)

# ------------------------------------------------------------------
# include owl-based implementation of owl_prime API. this will build a
# owl-based context that links itself to owl_prime and creates an actual,
# owl-based implementation of this API
# ------------------------------------------------------------------
add_subdirectory(owl-prime)


add_library(owl::owl_prime ALIAS owl_prime)
