# ======================================================================== #
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
add_definitions(${OWL_CXX_FLAGS})
embed_ptx(ptxCode deviceCode.cu)
add_executable(voxrender
  constants.h
  deviceCode.h
  hostCode.cpp
  readVoxOGT.cpp
  readVox.h
  resource.cpp
  ${ptxCode}
)
target_link_libraries(voxrender owl::owl ${OWL_VIEWER_LIBRARIES})
