add_library(impl
  impl.cpp
  )


add_executable(c-test
  c-test.c
  )
target_link_libraries(c-test
  impl
  )
# add_executable(test2
#   test2.c
#   )
# target_link_libraries(test2
#   impl
#   )



add_executable(cpp-test
  cpp-test.cpp
  )
target_link_libraries(cpp-test
  impl
  )


