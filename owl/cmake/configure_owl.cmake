# ======================================================================== #
# Copyright 2018-2020 Ingo Wald                                            #
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

# this helper script sets the following variables:
#
# OWL_INCLUDES - list of directories required to compile progs using owl
#
# OWL_LIBRARIES - list of libraries to link against when building owl programs

if(NOT OWL_INTERMEDIATE_CMAKE AND NOT OWL_MODERN_CMAKE)
  message(FATAL_ERROR "Please modify your parent project not to include "
    "configure_owl.cmake or any other internal CMake files from OWL. "
    "Exceptions: embed_ptx.cmake and get_gpu_arch.cmake, which you may "
    "need or want to use."
  )
endif()

set(OWL_LIBRARIES owl::owl)
