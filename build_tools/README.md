## How to build LLVM for tpp-mlir

The [instructions](https://github.com/plaidml/tpp-mlir?tab=readme-ov-file#how-to-build-llvm) to build LLVM are meant for cases where you only need to build once. 
In those cases, a sub module would be more helpful, but those cases are rare, with so many other cases where sub modules would be detrimental.

The main issues with using sub modules are:
* Cloning the LLVM repo takes a long time, and it's redundant if you already have it elsewhere (very likely).
* Building LLVM is an order of magnitude longer than tpp-mlir, and you probably already have the build elsewhere (personal or CI).
* CI would have one LLVM copy (source + build) per builder (different directories).
* Incompatible with [worktree](https://github.com/plaidml/tpp-mlir/wiki/Multiple-LLVM-Workspaces), which is the most sane way of developing in LLVM.

The main way we use `llvm_version.txt` are listed below.

### CI loop

In CI we have a separate LLVM build clones the PR's version and installs in a shared disk, where any tpp-mlir build (CI or not) can use.
This build has a _mutex_ so it only happens once per new hash. 
The install dir is the hash of the last commit, which means CI can run concurrently on builds that are trying new versions as well as old ones.

### Personal build in the cluster

The build and compute nodes have access to the shared drive, so if you have access to the CI cluster, you don't need to build LLVM at all.
Non-Intel folks can reproduce this on their local build server, using our CI scripts to build LLVM and install in a shared directory, to be used by multiple users.

### Local builds

If you are hacking on tpp-mlir, chances are you are also hacking on LLVM directly, and already have an LLVM tree (worktree even) elsewhere, and a build laying around.
You can create a branch for tpp-mlir and install in a local directory (with the commit hash as name) and set-up your tpp-mlir build to pick up on that.
This also helps when you go back and forth in tpp-mlir crossing LLVM version boundaries, there's no need to rebuild anything, as the previous hash install will still be there.

### Benchmark builds

If you're trying to reproduce our numbers from the paper, then you'll have a bit more work than recurring users to get everything built.
Luckily, we have written a [script](https://github.com/plaidml/tpp-mlir/tree/main/scripts/benchmarks) for that task, which will do most of the manual steps for you, and get the benchmarks running.
