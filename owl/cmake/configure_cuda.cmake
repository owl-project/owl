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

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include(get_gpu_arch)
get_gpu_arch(YOUR_GPU_ARCH)
message(STATUS "Your GPU architecture(s): ${YOUR_GPU_ARCH}")

if(OWL_INTERMEDIATE_CMAKE)
  if(OWL_CUDA_ARCH)
    string(STRIP "${OWL_CUDA_ARCH}" OWL_CUDA_ARCH)
    string(LENGTH "${OWL_CUDA_ARCH}" strlen)
    if((OWL_CUDA_ARCH MATCHES ";") OR (NOT strlen EQUAL 2 AND NOT strlen EQUAL 3) OR (NOT OWL_CUDA_ARCH GREATER 30))
      message(FATAL_ERROR "OWL_CUDA_ARCH is not a valid value \"${OWL_CUDA_ARCH}\", should be a number like 61 or 86!")
    endif()
  else()
    if(YOUR_GPU_ARCH)
      list(GET YOUR_GPU_ARCH 0 OWL_CUDA_ARCH)
    else()
      set(OWL_CUDA_ARCH 75)
    endif()
    message(STATUS "OWL_CUDA_ARCH not specified, defaulting to ${OWL_CUDA_ARCH}")
  endif()
  set(CMAKE_CUDA_FLAGS "-gencode arch=compute_${OWL_CUDA_ARCH},code=sm_${OWL_CUDA_ARCH} ${CMAKE_CUDA_FLAGS}")
  set(EMBED_PTX_ARCH "${OWL_CUDA_ARCH}")
elseif(OWL_MODERN_CMAKE)
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    if(YOUR_GPU_ARCH)
      set(CMAKE_CUDA_ARCHITECTURES "${YOUR_GPU_ARCH}")
      message(STATUS "CMAKE_CUDA_ARCHITECTURES not specified, defaulting to "
        "your GPU architecture(s) ${CMAKE_CUDA_ARCHITECTURES}"
      )
    else()
      message(FATAL_ERROR "CMAKE_CUDA_ARCHITECTURES not specified (required for "
        "modern CMake CUDA), and could not be detected automatically from your hardware"
      )
    endif()
  endif()
else()
  set(error_message "Please modify your parent project not to include "
    "configure_cuda.cmake or any other internal CMake files from OWL. "
    "Exceptions: embed_ptx.cmake and get_gpu_arch.cmake, which you may "
    "need or want to use."
  )
  if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(error_message "${error_message} Please see that file for info on how to specify the architecture.")
  endif()
  message(FATAL_ERROR "${error_message}")
endif()

link_libraries(cuda)
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}) # for C++
