# Benchmarking TPP-MLIR

This directory containst the script that we used in the paper to extract performance numbers.

It is the same process we use in our CI.

## Machines

For our paper, to achieve reproducible numbers, we have used the following AWS instances:

| Node | Vendor | Arch | Series | Cores | Threads | Memory | Disk |
| ---- | ------ | ---- | ------ | ----- | ------- | ------ | ---- |
| c6i.8xlarge | Intel | Ice Lake | Xeon 3 | 16 | 32 | 64GB | 300GB |
| c6a.8xlarge | AMD   | Milan | Zen 3 | 16 | 32 | 64GB | 300GB |
| c7i.8xlarge | Intel | Sapphire Rapids | Xeon 4 | 16 | 32 | 64GB | 300GB |
| c7a.8xlarge | AMD   | Genoa with | Zen 4 | 32 | 32 | 64GB | 300GB |
| c7g.8xlarge | Arm   | Graviton 3 | Neoverse V1 | 32 | 16 | 64GB | 300GB |

All instances above use the Amazon Linux, which is free on AWS.

## Benchmarks

The benchmarking script in this directory installs the necessary requirements, builds LLVM and TPP-MLIR and runs the following set of benchmarks:
 * Base: Comparison between `libxsmm-dnn` "hand-code" with `tpp-mir` compiler generated code.
 * PyTorch: Comparison between the compiler performance on Tensorflow-like generated IR with equivalent PyTorch extracted models.
 * OpenMP: Scalability analysis between `libxsmm-dnn` and `tpp-mlir` runs above on 2, 4, 8 and 16 threads.

## Execution

### Reserve the node

Use the AWS interface (web, cli) to reserve one of the nodes above. Make sure you can access it in a way to run the scripts (random public IP, elastic IP, console & password). Once the node is running, connect to the node and run the commands below.

### Commands

```sh
# First, identify the type of CPU/OS and its properties, it's good to keep a log to compare
$ lscpu && free -h && uname -a

# Then install git, and clone the repository
$ sudo dnf install -y git
$ git clone https://github.com/plaidml/tpp-mlir.git
$ cd tpp-mlir

# Finally run the install script
$ ./scripts/benchmarks/build_and_run.sh
```

It's recommended that you run the script on either `nohup`, `screen` or `tmux`, so that you can safely disconnect and reconnect later to gather the results.

## Results

Initially, the script will install packages, download and build LLVM, then use that LLVM to build TPP-MLIR and then use the benchmark driver to run the three benchmarks above.

The output is in `verbose` mode, to give an idea of progress and hint at failures if they do occur. The log messages are pre-fixed with a timestamp, the machine name and a message type (`DEBUG`, `INFO`). You should not see `ERROR`s but there may be `WARNING`s depending on the machine type. Some tests only run on `x86_64` while others only on `arm`.

Actual output does not have a log prefix and is in the following format:
```
Benchmark: NAME X
benchmark_x_dnn        :   104.446 gflops
benchmark_x_mlir       :   106.956 gflops
benchmark_x_torch      :    96.543 gflops
benchmark_x_omp_2_dnn  :   208.185 gflops
benchmark_x_omp_2_mlir :   212.223 gflops
```

Benchmarks with `dnn` in their names are results from the `libxsmm-dnn` executable and denote micro-kernels laid out by hand in C++ code, calling `libxsmm` directly.

Benchmarks with `mlir`in their names are executed by the compiler on some MLIR input. Of those, there are additional tags:
 * nohing: This is the default mode of the compiler and the _"fairest"_ comparison against `libxsmm-dnn`
 * `const`: Weights and biases are constant literals on the function, for compile time optimizations.
 * `args`: Weights and biases are arguments to the function, so harder to optimize.
 * `torch`: Models from PyTorch, instead of our `mlir-gen` _"Tensorflow-like"_ MLIR generator.
 * `omp_N`: OpenMP benchmarks on `N` threads.

## Troubleshooting

The script has been tested on developer machines and multiple AWS instances, so it should _"just work"_. However, it is not robust enough to run multiple times without failure. If something breaks in the middle, the worst case scenario is to fix the problem, remove everything and start over, until completion.

There are a number of things you can do to remove _"interim"_ state and re-start:
 * If package installation fail, there is no context yet, you can run the script again once the packages are installed correctly.
 * If the `build_llvm.sh` script has been called, it will create a directory `~/installs/llvm` before everything. So, if the LLVM build fails, you have to remove it before running it again. You'll also have to remove the zip file (`<hash>.zip`) and the build directory (`llvm`) in the source directory of tpp-mlir.
 * If the LLVM build succeeds, then the install will be detected and you won't need to worry about it again.
 * If the TPP-MLIR build fails, you can troubleshoot by following the main README and build by hand. The script will always try to build again, but because we use `ninja`, this will take less than a second on an already built directory.
 * The benchmark scripts can show warnings or errors on its log. If that happens, go to the `benchmarks` directory and follow instructions on how to use the `driver.py` script and adjust for your system.