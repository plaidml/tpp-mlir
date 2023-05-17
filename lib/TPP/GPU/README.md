# \[EXPERIMENTAL\] TPP GPU support

Experimental support for GPU code generation within TPP-MLIR.

_Note: the GPU support is disabled by default. It is an opt-in feature which can be enabled with CMake flag `-DTPP_GPU=ON`._

## LLVM with GPU support
Extra flags required:
```sh
-DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
-DCMAKE_CUDA_COMPILER=nvcc \
-DMLIR_ENABLE_CUDA_RUNNER=ON \
-DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
-DMLIR_ENABLE_SPIRV_CPU_RUNNER=ON \
-DMLIR_ENABLE_VULKAN_RUNNER=ON
```
Full build command:
```sh
cd llvm-project/build

cmake -G Ninja ../llvm \
   -DLLVM_ENABLE_PROJECTS=mlir \
   -DLLVM_BUILD_EXAMPLES=ON \
   -DLLVM_INSTALL_UTILS=ON \
   -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DLLVM_ENABLE_ASSERTIONS=ON \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DCMAKE_CUDA_COMPILER=nvcc \
   -DMLIR_ENABLE_CUDA_RUNNER=ON \
   -DMLIR_ENABLE_CUDA_CONVERSIONS=ON \
   -DMLIR_ENABLE_SPIRV_CPU_RUNNER=ON \
   -DMLIR_ENABLE_VULKAN_RUNNER=ON
```
## TPP MLIR with GPU support
Extra flags required:
```sh
-DTPP_GPU=ON
```
Full build command:
```sh
cd tpp-mlir/build

cmake -G Ninja .. \
   -DCMAKE_BUILD_TYPE=RelWithDebInfo \
   -DMLIR_DIR=$CUSTOM_LLVM_ROOT/lib/cmake/mlir \
   -DLLVM_EXTERNAL_LIT=$CUSTOM_LLVM_ROOT/bin/llvm-lit \
   -DCMAKE_C_COMPILER=clang \
   -DCMAKE_CXX_COMPILER=clang++ \
   -DTPP_GPU=ON
```

## CUDA setup
- Install [CUDA on WSL 2](https://docs.nvidia.com/cuda/wsl-user-guide/index.html#getting-started-with-cuda-on-wsl)
    1. **HOST** - install newest [NVIDIA drivers on host](https://www.nvidia.com/Download/index.aspx)
    2. **WSL** - remove the old GPG key: `sudo apt-key del 7fa2af80`
    3. **WSL** - follow instructions to install [x86 CUDA Toolkit](https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local)
- [CUDA samples](https://github.com/nvidia/cuda-samples) - useful to test CUDA installation

## Troubleshooting
- CUDA driver vs runtime version mismatch:
    - check if CUDA driver and cuda-toolkit versions are the same:
    ```
    $ nvidia-smi
    $ dpkg -l | grep cuda-toolkit
    ```
    - thread with more details - [link](https://forums.developer.nvidia.com/t/cuda-driver-version-is-insufficient-for-cuda-runtime-version-wsl2-ubuntu-18-04/178720/11)
- MLIR CUDA_ERROR_ILLEGAL_ADDRESS bug - [link](https://bugs.llvm.org/show_bug.cgi?id=51107)
