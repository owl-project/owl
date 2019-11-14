add_library(impl
  impl.cpp
  )
add_executable(c-test
  c-test.c
  )
target_link_libraries(c-test
  impl
  )


