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

# this helper script sets the following variables:
#
# OWL_INCLUDES - list of directories required to compile progs using owl
#
# OWL_LL_LIBRARIES - list of library names required to build apps
# using (only) the ll layer
#
# OWL_NG_LIBRARIES - list of library names required to build apps
# using the node graph layer

set(OWL_INCLUDES
  # owl needs cuda:
  ${CUDA_TOOLKIT_ROOT_DIR}/include
  # owl needs optix:
  ${OptiX_INCLUDE}
  # public API
  ${owl_dir}/owl/include
  # device API and common currently still include non-public header files
  ${owl_dir}/
  )
set(OWL_LL_LIBRARIES
  llowl
  )
set(OWL_NG_LIBRARIES
  llowl-static
  owl-ng
  )
# if in doubt, use both:
set(OWL_LIBRARIES
  ${OWL_LL_LIBRARIES}
  ${OWL_NG_LIBRARIES}
  )
