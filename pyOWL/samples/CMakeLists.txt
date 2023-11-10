function(pyowl_compile_ptx output_ptx input_cu)
  add_library(${output_ptx} OBJECT)
  target_sources(${output_ptx} PRIVATE ${COMPILE_PTX_SOURCES})
#  target_link_libraries(${output_ptx} PRIVATE owl::owl)
  set_property(TARGET ${output_ptx} PROPERTY CUDA_PTX_COMPILATION ON)
  set_property(TARGET ${output_ptx} PROPERTY CUDA_ARCHITECTURES OFF)
  target_compile_options(${output_ptx} PRIVATE "-lineinfo")
  target_sources(${output_ptx} PRIVATE ${input_cu})
  add_custom_target(build_${output_ptx} ALL DEPENDS ${output_ptx}
    COMMAND  cp "$<TARGET_OBJECTS:${output_ptx}>" ${PROJECT_BINARY_DIR}/${output_ptx}
    BYPRODUCTS ${PROJECT_BINARY_DIR}/${output_ptx}
    VERBATIM)
endfunction()

# ------------------------------------------------------------------
pyowl_compile_ptx(cmd01-simpleTriangles.ptx cmd01-simpleTriangles/deviceCode.cu)
target_link_libraries(cmd01-simpleTriangles.ptx PUBLIC owl::owl)

# ------------------------------------------------------------------
pyowl_compile_ptx(cmd05-rtow.ptx cmd05-rtow/deviceCode.cu)
target_link_libraries(cmd05-rtow.ptx PUBLIC owl::owl)

