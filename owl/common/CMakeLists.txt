# ======================================================================== #
# Copyright 2019-2020 Ingo Wald                                            #
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

# this library is mostly header-only, but let's list it anyway -
# including all the header fiels - to make visual studio happy
add_library(owl-common STATIC

  owl-common.cpp
  
  ../include/owl/common/owl-common.h
  ../include/owl/common/math/vec.h
  ../include/owl/common/math/box.h
  ../include/owl/common/math/LinearSpace.h
  ../include/owl/common/math/AffineSpace.h

  parallel/parallel_for.h
  
  cmake/configure_optix.cmake
  cmake/configure_cuda.cmake
  cmake/configure_build_type.cmake
  )
