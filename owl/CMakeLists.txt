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

add_definitions(${OWL_CXX_FLAGS})

# owl-common library (math stuff, utils, etc; mostly header-only)
add_subdirectory(common)

# low-level interface
add_subdirectory(ll)

# node graph layer API
add_subdirectory(api)

# -------------------------------------------------------
# export those back to our parent:
# -------------------------------------------------------
set(OWL_INCLUDES ${OWL_INCLUDES} PARENT_SCOPE)
set(OWL_LIBRARIES ${OWL_LIBRARIES} PARENT_SCOPE)
