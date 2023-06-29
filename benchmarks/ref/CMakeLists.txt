get_filename_component(BENCH_REF_INCLUDE_DIR
                       ./include
                       ABSOLUTE)

if (TPP_GPU)
    add_subdirectory(GPU)
endif()
