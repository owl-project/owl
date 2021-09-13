# Copyright (C) 2021 NVIDIA Corporation
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
# get_gpu_arch.cmake
# ^^^^^^^^^^^^^^^^^^
# 
# Returns a list of CUDA-compatible GPU architectures on the current machine,
# in increasing order of compute capability.
# Examples:
#   52      (if you have a GTX 980)
#   61;86   (if you have an RTX 3080 and a GTX 1080)
# If the query fails or there are no CUDA-compatible GPUs, returns an empty string.
# The return value is set in a variable named [whatever you pass in OUT_VAR].

function(get_gpu_arch OUT_VAR)
  set(test_prog_source "#include <stdio.h>
int main(){
    cudaDeviceProp p;
    int n,g=0,e=cudaGetDeviceCount(&n);
    for(;g<n&&!e;++g){
        if((e=cudaGetDeviceProperties(&p,g)))break;
        printf(\"%d%d%s\",p.major,p.minor,g<n-1?\";\":\"\");
    }
    printf(\"%s\\n\",e?\"Error\":!n?\"0\":\"\");
    return e;
}
")
  set(sourcefile "${CMAKE_CURRENT_BINARY_DIR}/get_gpu_arch.cu")
  file(WRITE ${sourcefile} "${test_prog_source}")
  set(cudaflags "")
  if(NOT WIN32 AND CMAKE_CUDA_HOST_COMPILER)
    set(cudaflags "-ccbin ${CMAKE_CUDA_HOST_COMPILER}")
  endif()
  set(binfile "${CMAKE_CURRENT_BINARY_DIR}/get_gpu_arch")
  if(WIN32)
    set(binfile "${binfile}.exe")
  endif()
  if(WIN32)
    message(STATUS "Compile command: ${CMAKE_CUDA_COMPILER} -ccbin ${CMAKE_CXX_COMPILER} ${sourcefile} -o ${binfile}")
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} -ccbin ${CMAKE_CXX_COMPILER} ${sourcefile} -o ${binfile}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE compile_result
    )
  else()
    execute_process(
      COMMAND ${CMAKE_CUDA_COMPILER} ${cudaflags} ${sourcefile} -o ${binfile}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      RESULT_VARIABLE compile_result
    )
  endif()
  set(final_result "")
  if(NOT ${compile_result} EQUAL "0")
    message(STATUS "get_gpu_arch compile failed, could not determine your GPU architecture")
  else()
    execute_process(
      COMMAND ${binfile}
      WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
      OUTPUT_VARIABLE output
      RESULT_VARIABLE result
    )
    string(STRIP "${output}" output)
    if(NOT result EQUAL "0" OR output MATCHES "Error")
      message(STATUS "get_gpu_arch failed, could not determine your GPU architecture")
    elseif(output EQUAL "0")
      message(STATUS "No CUDA capable GPUs found")
    else()
      list(SORT output)
      set(final_result "${output}")
    endif()
  endif()
  #message(STATUS "final_result: ${final_result}")
  set(${OUT_VAR} "${final_result}" PARENT_SCOPE)
endfunction()
