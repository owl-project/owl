## ======================================================================== ##
## Copyright 2018-2019 Ingo Wald                                            ##
##                                                                          ##
## Licensed under the Apache License, Version 2.0 (the "License");          ##
## you may not use this file except in compliance with the License.         ##
## You may obtain a copy of the License at                                  ##
##                                                                          ##
##     http://www.apache.org/licenses/LICENSE-2.0                           ##
##                                                                          ##
## Unless required by applicable law or agreed to in writing, software      ##
## distributed under the License is distributed on an "AS IS" BASIS,        ##
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. ##
## See the License for the specific language governing permissions and      ##
## limitations under the License.                                           ##
## ======================================================================== ##

if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON) # moved from main CMakeLists.txt

if(OWL_DEPRECATED_CMAKE)
  set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
  #find_package(CUDA REQUIRED)
  find_package(CUDA)

  if (CUDA_FOUND)
    include_directories(${CUDA_TOOLKIT_INCLUDE})
    set(CUDA_SEPARABLE_COMPILATION ON)
  endif()
else()
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

  if(OWL_INTERMEDIATE_CMAKE)
    if(NOT OWL_CUDA_ARCH)
      message(WARNING "OWL_CUDA_ARCH not set, defaulting to 75")
      set(OWL_CUDA_ARCH 75)
    endif()
    string(STRIP "${OWL_CUDA_ARCH}" OWL_CUDA_ARCH)
    string(LENGTH "${OWL_CUDA_ARCH}" strlen)
    if((NOT strlen EQUAL 2) OR (NOT OWL_CUDA_ARCH GREATER 30))
      message(FATAL_ERROR "OWL_CUDA_ARCH is not a valid value \"${OWL_CUDA_ARCH}\", should be a number like 61 or 86!")
    endif()
    set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${OWL_CUDA_ARCH},code=sm_${OWL_CUDA_ARCH} ${CMAKE_CUDA_FLAGS}")
    set(EMBED_PTX_ARCH "${OWL_CUDA_ARCH}")
  else()
    if(NOT CMAKE_CUDA_ARCHITECTURES)
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES must be set!")
    endif()
  endif()

  link_libraries(cuda)
  include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # for C++
endif()
