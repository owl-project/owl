# ======================================================================== #
# Copyright 2019 Ingo Wald                                                 #
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

# moved "shared" device code to a library, else cmake scripts get
# confused about using the same file in multiple binaries.
cuda_compile_and_embed(ptxCode
  deviceCode.cu
  )
add_library(ll07-deviceCode STATIC
  ${ptxCode}
  )

include_directories(${PROJECT_SOURCE_DIR}/owl)
add_executable(ll07-groupOfGroups
  hostCode.cpp
  )
target_link_libraries(ll07-groupOfGroups
  ${OWL_LL_LIBRARIES}
  ll07-deviceCode
  )
add_test(ll07-groupOfGroups
  ${CMAKE_BINARY_DIR}/ll07-groupOfGroups
  )



foreach(test 1;2)
  add_executable(ll07-groupOfGroups-variant${test} 
    hostCode.cpp
    )
  target_link_libraries(ll07-groupOfGroups-variant${test}
    ${OWL_LL_LIBRARIES}
    ll07-deviceCode
    )
  target_compile_definitions(ll07-groupOfGroups-variant${test}
    PRIVATE
    -DTEST_CASE=${test})
  add_test(ll07-groupOfGroups-variant${test}
    ${CMAKE_BINARY_DIR}/ll07-groupOfGroups-variant${test}
    )
endforeach()



