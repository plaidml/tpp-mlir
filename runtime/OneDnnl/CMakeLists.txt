add_mlir_library(tpp_dnnl_runner_utils
  SHARED
  OneDnnlRunnerUtils.cpp
  ../PerfRunnerUtils.cpp

  LINK_LIBS PRIVATE
  dnnl
  )

set_property(TARGET tpp_dnnl_runner_utils PROPERTY CXX_STANDARD 11)
target_compile_definitions(tpp_dnnl_runner_utils PRIVATE mlir_c_runner_utils_EXPORTS)
