option(USE_OpenMP "Use OpenMP" ON)

# We don't want GOMP because its performance sinks for large core count, so we force libomp
# This finds the library path from the system's clang for OpenMP
#
# On Fedora, it's at the same place as others, so we don't need to look elsewhere
# On Ubuntu, it's in /usr/lib/llvm-${version}, so find_package finds GOMP for GCC instead.
execute_process (
    COMMAND bash -c "for lib in $(clang -lomp -### 2>&1); do echo $lib | grep -o \"\\/.*llvm.*\\w\"; done"
    OUTPUT_VARIABLE LLVM_OMP_PATH
)
# Only if we found an "llvm" path that we need to add
if (LLVM_OMP_PATH)
  set(CMAKE_PREFIX_PATH ${LLVM_OMP_PATH})
endif()

if(USE_OpenMP)
  find_package(OpenMP)
  if(OPENMP_FOUND)
    message(STATUS "OpenMP found")
  else()
    message(FATAL_ERROR "OpenMP required but not found")
  endif()
endif()
