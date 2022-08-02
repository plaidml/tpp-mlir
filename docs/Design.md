Design decision.

**TPP dialect**. TPP stands for tensor processing primitives, and it is our new
"ISA". TPP is the end IR we want to lower before using a custom assembler to
achieve high-performance code (the assembler is LIBXSMM). We position TPP after
linalg for the following reasons:

- Most of the ML frontends are (or will) lower to linalg. Therefore we can
  funnel all the compilation flows to TPP.
- From an abstraction perspective, it makes sense to position TPP below linalg
  as TPP is our endpoint. You can think of TPP as micro-kernel for Linalg. At
the same time, we want to use linalg to perform most of the optimization we are
interested in, and the lower to TPP. 
- TPP operates at memref level as we can preserve parallelism when lowering
  linalg on buffers (see all the discussion on the distructive update at tensor
level -
https://discourse.llvm.org/t/how-to-represent-linalg-on-subtensors/4531). We
also want to reuse bufferization to allocate/deallocate memory. 

*Some caveats*: Since we are targeting a "library-call" kind of computation, we
need to understand what the linalg kernel is computing. Unfortunately, this
information is lost if we generalize the operations (i.e., linalg.matmul to
linalg.generic). So, are we missing an abstraction level? Maybe yes, or maybe
not; the answer is still unclear (see:
https://discourse.llvm.org/t/rfc-primitive-ops-add-mapop-reductionop-transposeop-broadcastop-to-linalg/64184).

Optimization we want to do at Linalg level:

- Element-wise fusion and produce-consumer fusion

- Blocking (relayout the tensor to reduce strides and avoid cache conflicts)

- Logical tiling to map n-d tensor to 2-d tensor before lower to TPP 

**XSMM dialect** Interface dialect to the assembler. Consists of dispatc
operations and invoke operations.
