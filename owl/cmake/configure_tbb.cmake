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

# this should't be a public option, rather something the parent project can set
# (if used as a submodule)
#OPTION(OWL_DISABLE_TBB "DISABLE TBB in OWL, even if it could be found" OFF)
if (OWL_DISABLE_TBB)
  set(OWL_USE_TBB OFF)
else()
 if (WIN32)
   # OFF by default under windows; windows probably doesnt' have it, 
   # nor is there an easy way to install it
   OPTION(OWL_USE_TBB "Use TBB to parallelize host-side code?" OFF)
 else()
   # on by default, because even if not installed it's trivial
   # to apt/yum install
  OPTION(OWL_USE_TBB "Use TBB to parallelize host-side code?" ON)
  endif()
endif()

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

if (OWL_USE_TBB AND (NOT OWL_DISABLE_TBB))

  #
  # first try find_package in CONFIG mode, it will be catched by OneTBB
  # if OneTBB is not avaible, then use the old macro
  #
  find_package(TBB CONFIG)
  if (NOT TBB_FOUND)
    find_package(TBB)
  endif()

  if (TBB_FOUND)
    #    include_directories(${TBB_INCLUDE_DIR})
#    set(OWL_CXX_FLAGS "${OWL_CXX_FLAGS} -DOWL_HAVE_TBB=1")
    message(STATUS "#owl.cmake: found TBB, in include dir ${TBB_INCLUDE_DIR}")
    set(OWL_HAVE_TBB ON)
  else()
    message(STATUS "#owl.cmake: TBB not found; falling back to serial execution of owl::parallel_for")
    set(OWL_HAVE_TBB OFF)
  endif()
else()
  set(OWL_HAVE_TBB OFF)
endif()
