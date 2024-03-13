if (POLICY CMP0077)
  cmake_policy(SET CMP0077 NEW)
endif()

set(CUDA_USE_STATIC_CUDA_RUNTIME ON)
#find_package(CUDA REQUIRED)
#find_package(CUDA)

if (CUDA_FOUND)
  include_directories(${CUDA_TOOLKIT_INCLUDE})
  
  set(CUDA_SEPARABLE_COMPILATION ON)
endif()

macro(cuda_compile_to_ptx output_file cuda_file)
  cuda_compile_ptx(ptx_files
    ${cuda_file}
    )
  list(GET ptx_files 0 ptx_file)
  add_custom_target(${output_file} ALL
    COMMAND ${CMAKE_COMMAND} -E copy ${ptx_file} ${output_file}
    DEPENDS ${cuda_file} ${ptx_file}
    )
endmacro()
