# Copyright (C) 2020 NVIDIA Corporation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# embed_ptx.cmake
# ^^^^^^^^^^^^^^^
# Compiles a CUDA source file to PTX, and creates a C file containing a
# character array of the text of the PTX.
# 
# Architectures and flags: If EMBED_PTX_ARCH is set (should be set to a value
# like e.g. 61 or 86), generates code for just this architecture. Otherwise,
# if CMAKE_CUDA_ARCHITECTURES is set, uses the oldest (lowest-numbered)
# architecture set in CMAKE_CUDA_ARCHITECTURES. Otherwise, it will issue a
# warning and use the NVCC default. 
# Flags are propagated from CMAKE_CUDA_FLAGS, and includes are also propagated
# from the current source directory.
# 
# @param c_embed_name This argument serves two functions.
# First, it is the name of the character array containing the PTX source.
# Second, a CMake variable by this name will be set by the function, containing
# the path to the generated C file.
# 
# @param cu_file The literal name of the input CUDA file, relative to the
# current source directory.
# 
# Example:
# embed_ptx(myPtxProgram my_program.cu)
# add_executable(foo a.c b.cu c.cpp ${myPtxProgram})
# 
# WARNING: embed_ptx MUST be called in the same directory (CMakeLists.txt) as
# the add_executable or add_library call which ${myPtxProgram} is being included
# in. Adding it later, e.g. with target_sources from a child directory, will NOT
# work. This is a limitation of CMake, see https://stackoverflow.com/questions/
# 57824263/cmake-doesnt-recognize-custom-command-as-valid-source
# 
# WARNING: Your CMake top-level project MUST include the C language, otherwise
# CMake will never emit rules to build the generated C file into an object file
# and you will get "undefined reference to myPtxProgram" or "unresolved external
# symbol myPtxProgram" from the linker.

# Replaces list(TRANSFORM ... PREPEND ...) which isn't available in 3.8.
# From https://github.com/flutter/flutter/pull/57515/files
# Copyright 2014 The Flutter Authors. All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above
#       copyright notice, this list of conditions and the following
#       disclaimer in the documentation and/or other materials provided
#       with the distribution.
#     * Neither the name of Google Inc. nor the names of its
#       contributors may be used to endorse or promote products derived
#       from this software without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
# ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON
# ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
function(list_prepend LIST_NAME PREFIX)
  set(NEW_LIST "")
  foreach(element ${${LIST_NAME}})
    list(APPEND NEW_LIST "${PREFIX}${element}")
  endforeach(element)
  set(${LIST_NAME} "${NEW_LIST}" PARENT_SCOPE)
endfunction()

# Replaces list(TRANSFORM ... REPLACE ...)
function(list_replace LIST_NAME FINDREGEX REPLACEWITH)
  set(NEW_LIST "")
  foreach(element ${${LIST_NAME}})
    string(REGEX REPLACE "${FINDREGEX}" "${REPLACEWITH}" elementout "${element}")
    list(APPEND NEW_LIST "${elementout}")
  endforeach(element)
  set(${LIST_NAME} "${NEW_LIST}" PARENT_SCOPE)
endfunction()

function(embed_ptx c_embed_name cu_file)
  # Initial setup
  set(ptxfile "${CMAKE_CURRENT_BINARY_DIR}/${c_embed_name}.ptx")
  set(embedfile "${CMAKE_CURRENT_BINARY_DIR}/${c_embed_name}.c")
  file(RELATIVE_PATH cu_file_frombin ${CMAKE_CURRENT_BINARY_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/${cu_file})
  # Find bin2c
  if(NOT BIN2C)
    get_filename_component(cudabindir ${CMAKE_CUDA_COMPILER} DIRECTORY)
    if(WIN32)
      set(BIN2C "${cudabindir}/bin2c.exe")
    else()
      set(BIN2C "${cudabindir}/bin2c")
    endif()
    if(NOT EXISTS ${BIN2C})
      message(FATAL_ERROR "Could not find bin2c at ${BIN2C}!")
    endif()
  endif()
  # Architecture
  if(EMBED_PTX_ARCH)
    set(archflags "-gencode arch=compute_${EMBED_PTX_ARCH},code=compute_${EMBED_PTX_ARCH}")
  elseif(CMAKE_CUDA_ARCHITECTURES)
    set(archflags ${CMAKE_CUDA_ARCHITECTURES})
    list(SORT archflags CASE INSENSITIVE)
    list(GET archflags 0 archflags)
    string(REPLACE "-real" "" archflags ${archflags})
    string(REPLACE "-virtual" "" archflags ${archflags})
    set(archflags "-gencode arch=compute_${archflags},code=compute_${archflags}")
  else()
    set(archflags "")
    message(WARNING "embed_ptx: no EMBED_PTX_ARCH or CMAKE_CUDA_ARCHITECTURES specified, using compiler default")
  endif()
  # Flags manipulation
  string(REGEX REPLACE " -gencode arch=[A-Za-z0-9_,=]+" "" manualcudaflags "${CMAKE_CUDA_FLAGS}")
  while(manualcudaflags MATCHES ".*\".*")
    if(NOT manualcudaflags MATCHES ".*\".*\".*")
      message(FATAL_ERROR "CUDA flags for embed PTX only contain one quotation mark: ${manualcudaflags}")
    endif()
    string(REGEX REPLACE "[^\"]*\"([^\"]*)\".*" "\\1" quotedcontents "${manualcudaflags}")
    string(REGEX REPLACE "[^\"]* ([-A-Za-z]*)=? ?\".*" "\\1" quoteprefix "${manualcudaflags}")
    string(REGEX REPLACE "([^\"]*) [-A-Za-z]*=? ?\".*" "\\1" beforequotes "${manualcudaflags}")
    string(REGEX REPLACE "[^\"]*\"[^\"]*\"(.*)" "\\1" afterquotes "${manualcudaflags}")
    string(STRIP "${quotedcontents}" quotedcontents)
    string(STRIP "${quoteprefix}" quoteprefix)
    string(REPLACE " " ";" quotedargs "${quotedcontents}")
    list_prepend(quotedargs "${quoteprefix} ")
    string(REPLACE ";" " " quotedargs "${quotedargs}")
    set(manualcudaflags "${beforequotes} ${quotedargs} ${afterquotes}")
  endwhile()
  # Includes manipulation
  get_property(includes DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY INCLUDE_DIRECTORIES)
  if(WIN32)
    list_replace(includes " " "\\\\ ")
  endif()
  list_prepend(includes "-I ")
  string(REPLACE ";" " " includes "${includes}")
  # Final flags manipulation
  set(allcudaflags "${manualcudaflags} ${archflags} ${includes}")
  #message(STATUS "All CUDA flags for embed_ptx of ${cu_file}:\n${allcudaflags}")
  separate_arguments(allcudaflags)
  #message(STATUS "embed_ptx build command 1: ${CMAKE_CUDA_COMPILER} ${allcudaflags} -ptx ${cu_file_frombin} -o ${ptxfile}")
  #message(STATUS "embed_ptx build command 2: ${BIN2C} -c --padd 0 --type char --name ${c_embed_name} ${ptxfile} > ${embedfile}")
  # Ending
  add_custom_command(
    OUTPUT ${embedfile}
    COMMAND echo ${CMAKE_CUDA_COMPILER} ${allcudaflags} -ptx ${cu_file_frombin} -o ${ptxfile}
    COMMAND ${CMAKE_CUDA_COMPILER} ${allcudaflags} -ptx ${cu_file_frombin} -o ${ptxfile}
    COMMAND echo ${BIN2C} -c --padd 0 --type char --name ${c_embed_name} ${ptxfile} > ${embedfile}
    COMMAND ${BIN2C} -c --padd 0 --type char --name ${c_embed_name} ${ptxfile} > ${embedfile}
    DEPENDS ${cu_file}
    COMMENT "Building embedded PTX source ${embedfile}"
    VERBATIM
  )
  set(${c_embed_name} ${embedfile} PARENT_SCOPE)
endfunction()
