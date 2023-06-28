get_filename_component(BENCH_REF_INCLUDE_DIR
                       ./include
                       ABSOLUTE)

add_subdirectory(matmul)
add_subdirectory(GPU)
