if (NOT TPP_INSIDE_IREE)
  add_mlir_library(tpp_c_runner_utils
    SHARED
    XsmmRunnerUtils.cpp
    PerfRunnerUtils.cpp

    LINK_LIBS PUBLIC
    xsmm
  )
  set_property(TARGET tpp_c_runner_utils PROPERTY CXX_STANDARD 11)
  target_compile_definitions(tpp_c_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)
else()
  add_library(tpp_c_runner_utils
    SHARED
    XsmmRunnerUtils.cpp
    PerfRunnerUtils.cpp
  )
  target_link_libraries(tpp_c_runner_utils xsmm)
endif()
