if (NOT TPP_INSIDE_IREE)
  add_mlir_library(standalone_c_runner_utils
    SHARED
    XsmmRunnerUtils.cpp

    LINK_LIBS PUBLIC
    xsmm
  )
  set_property(TARGET standalone_c_runner_utils PROPERTY CXX_STANDARD 11)
  target_compile_definitions(standalone_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)
else()
  add_library(standalone_c_runner_utils
    STATIC
    XsmmRunnerUtils.cpp
  )
  target_link_libraries(standalone_c_runner_utils xsmm)
endif()
