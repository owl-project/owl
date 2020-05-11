# ======================================================================== #
# Copyright 2020 Ingo Wald                                                 #
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

# configures the glfw library

include(configure_glfw)
if (OWL_HAVE_GLFW)
  set(OWL_VIEWER_INCLUDES
    ${glfw3_DIR}
    ${owl_dir}/samples/common/
    )
  set(OWL_HAVE_VIEWER ON)
else()
  set(OWL_HAVE_VIEWER OFF)
endif()
