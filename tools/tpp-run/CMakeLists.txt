get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)

if(USE_OneDNN)
  set(ONEDNN_LIBS "tpp_dnnl_runner_utils")
  set(ONEDNN_LIBS_INCL "-ltpp_dnnl_runner_utils")
endif()

if(USE_OpenMP)
  set(OPENMP_LIBS_INCL "-lomp")
endif()

set(LIBS
        ${dialect_libs}
        ${conversion_libs}
        MLIRAnalysis
        MLIRExecutionEngine
        MLIRIR
        MLIRJitRunner
        MLIRLLVMDialect
        MLIRLLVMToLLVMIRTranslation
        MLIRToLLVMIRTranslationRegistration
        MLIRParser
        MLIRTargetLLVMIRExport
        MLIRSupport
        MLIROptLib
        MLIRTPP
        tpp_xsmm_runner_utils
        ${ONEDNN_LIBS}
        )

set(LLVM_LINK_COMPONENTS
  Core
  Support
  nativecodegen
  native
  )

add_llvm_executable(tpp-run
  MLIRBench.cpp
  tpp-run.cpp)

llvm_update_compile_flags(tpp-run)

target_link_libraries(tpp-run PRIVATE ${LIBS})

if (TPP_GPU MATCHES "cuda")
  set(TPP_GPU_LINK_FLAGS
      ${TPP_GPU_LINK_FLAGS}
      -lmlir_cuda_runtime
    )
endif()

if (TPP_GPU MATCHES "vulkan")
  set(TPP_GPU_LINK_FLAGS
      ${TPP_GPU_LINK_FLAGS}
      -lvulkan-runtime-wrappers
    )
endif()

message(STATUS "TPP libraries at: ${CMAKE_BINARY_DIR}/lib")
message(STATUS "MLIR libraries at: ${LLVM_LIBRARY_DIR}")
# Add TPP/MLIR library path so we don't need --shared-libs on every call
# TODO: I don't understand... they are already in LIBS above.
# also --no-as-needed is not available on MAC and is repeated.
target_link_options(tpp-run PRIVATE
  -Wl,--no-as-needed
  -L${CMAKE_BINARY_DIR}/lib
  -ltpp_xsmm_runner_utils
  ${ONEDNN_LIBS_INCL}
  -L${LLVM_LIBRARY_DIR}
  -lmlir_c_runner_utils
  -lmlir_runner_utils
  -lmlir_async_runtime
  ${TPP_GPU_LINK_FLAGS}
  ${OPENMP_LIBS_INCL}
  -Wl,--as-needed
)

install(TARGETS tpp-run RUNTIME DESTINATION bin)
