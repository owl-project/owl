#
# Copyright (c) 2018 NVIDIA CORPORATION. All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#  * Redistributions of source code must retain the above copyright
#    notice, this list of conditions and the following disclaimer.
#  * Redistributions in binary form must reproduce the above copyright
#    notice, this list of conditions and the following disclaimer in the
#    documentation and/or other materials provided with the distribution.
#  * Neither the name of NVIDIA CORPORATION nor the names of its
#    contributors may be used to endorse or promote products derived
#    from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS ``AS IS'' AND ANY
# EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
# PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY
# OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#

# Locate the OptiX distribution. As of OptiX 7 we only need to find the header file

if (NOT DEFINED OptiX_INCLUDE)
  # optix include not defined on cmdline, try to find ....
  if (DEFINED(OptiX_INSTALL_DIR))
    message(STATUS "going to look for OptiX_INCLUDE in OptiX_INSTALL_DIR=${OptiX_INSTALL_DIR}")
    set(OptiX_INCLUDE_SEARCH_PATH ${OptiX_INSTALL_DIR}/include)
  elseif (EXISTS "$ENV{OptiX_INSTALL_DIR}")
    set(OptiX_INCLUDE_SEARCH_PATH $ENV{OptiX_INSTALL_DIR}/include)
    message(STATUS "going to look for OptiX_INCLUDE in OptiX_INSTALL_DIR=$ENV{OptiX_INSTALL_DIR} (from environment variable)")
  else ()
    message(STATUS "OptiX_INSTALL_DIR not defined in either cmake or environment.... please define this, or set OptiX_INCLUDE manually in your cmake GUI")
    set(Optix_INCLUDE_SEARCH_PATH)
  endif ()
  find_path(OptiX_INCLUDE
    NAMES optix.h
    PATHS "${OptiX_INCLUDE_SEARCH_PATH}"
    #NO_DEFAULT_PATH
  )

# Check to make sure we found what we were looking for
function(OptiX_report_error error_message required)
  if(OptiX_FIND_REQUIRED AND required)
    message(FATAL_ERROR "${error_message}")
  else()
    if(NOT OptiX_FIND_QUIETLY)
      message(STATUS "${error_message}")
    endif(NOT OptiX_FIND_QUIETLY)
  endif()
endfunction()

if(NOT OptiX_INCLUDE)
  OptiX_report_error("OptiX headers (optix.h and friends) not found.  Please locate before proceeding." TRUE)
endif()

endif()
