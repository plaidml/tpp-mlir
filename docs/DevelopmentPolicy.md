# Development Policy

This is a policy that we should follow in this project to allow us to work together.
For guidelines on how to create issues or pull requests, please check the  [contributions](../CONTRIBUTING.md) guidelines.

This is a project based on LLVM, so we should also follow their [developer policy](https://llvm.org/docs/DeveloperPolicy.html), to allow us to upstream code more easily.

## Scope

This is a research project for designing compiler passes to make use of micro-kernels for performing execution on diverse hardware.
These passes can be embedded or used by other frameworks (ex. IREE, OpenXLA, PyTorch, etc) as well as stand-alone.
As such, we can't depend on any particular framework but we should design them to be used by any of them.

In that sense, we need independent methods for acquiring input IR as well as being able to use any runtime to execute the kernels we optimise.
And we should consider all options when adding or changing functionality, preferrably through a stable CI environment.

So, the policy in this document is written to achieve those goals, while still allowing us to progress on our core mission and follow the LLVM project closely on trunk.

## Stability

Even though this isn't a product, nor it aims to ever be one, we still need some stability guarantees.
The main reasons for stability are:
 * To allow multiple developers to contribute to the project without fear of constant rewrites, rebases and rework.
 * To ensure _forward progress_ as our basic mantra. We don't break existing features in order to add new ones.
 * To collaborate with other projects seamlessly, including frameworks (IREE, OpenXLA) and runtime libraries (libxsmm, OneDNN).
 * To avoid performance regressions, which are often brought to by changes in upstream LLVM.

## Progress

We need to progress on our core value, compiler transformations, while still guaranteeing our previous work hasn't regressed.
But we also don't want to spend our entire time fixing LLVM bugs, so we need to make sure our contributions are effective and permanent.

Local fixes to upstream bugs are faster, but can often be broken again. So if the problem is upstream, we must fix upstream, and make sure the intention of the fix is retained by future changes in the same areas (consensus, tests).

But upstream fixes can take time, and a different shape than we need downstream, so we adapt.
The best strategy, as always, is a mix of downstream + upstream process, where everything is aimed at upstream, but some things live downstream for a while before acquiring the right shape.

## Testing & CI

For each feature we add we should add a set of tests that cover the main cases, and edge cases of the feature. Often those are one and the same, but not always. So if there are valid ranges or specific conditions that are not allowed, we should add negative tests to make sure we get those errors.

These tests can be either IR-to-IR, CHECKing output for specific instructions generated, or execution tests, CHECKing output for valid tensor elements, etc.

### Regression Tests

Every time we find a bug or a performance regression, we should add a regression test to make sure the scenario doesn't come back later. For example, checking that extra copies aren't introduced, or that a specific fusion has been done. Those are the same as above, but in response to a regression.

### Integration Tests

These tests check more than one step in the process, testing the _integration_ of the different steps. IR generation _and_ compiler passes, or compiler passes _and_ execution, etc.

These should be complementary to basic tests, and doesn't need to care about individual steps, just make sure that the connection between them works fine.

### Benchmarks

There are two main types of benchmarks:
 * **Micro-benchmarks**: small kernels of hand-crafted code that represents a particular computation pattern. We want those to measure just the execution of the kernels in question with some statistical harness.
 * **End-to-end**: whole models / programs, executing what end-users would do, composing the kernels above into a single program, measuring the end-to-end latency and bandwidth with some harness to prepare the environment and measure the right things.

We want enough of those benchmarks to run on continuous integration, preferably some of them as a pre-merge check, to guarantee forward progress, making sure we fix performance regressions before it's too late.

## Fixing Bugs

When we find bugs in the code, and it's not part of the current feature/fix we're working on, we should create an issue (see [contributing](../CONTRIBUTING.md)) to allow other developers to help.

It may be that they know how to fix, or can fix faster than you, or even that there is already a fix elsewhere. Otherwise, once you finish your current task, it should be easier to look at the existing issues and prioritise.

If a pull request introduces bugs or performance regressions, it should be fixed before merging. But if the request is for merging LLVM, it may not be that simple.

Sometimes we need a new LLVM for features we're working on, or to fix other bugs that have shown up. Sometimes we need it to pull changes that we made ourselves to LLVM, and small performance regressions may be overlooked in order to fix a bug or a bigger performance regression.

However, the rule above still applies: If there's any bug or performance regression, we need to create an issue with all the details, so that we can priotisie it before the next feature/bug-fix is taken by any member of the team.

This ensures _forward progress_ can only temporarily be blocked, with a clear path forward.

### Fixing Upstream Bugs Downstream

Some LLVM changes may break our stuff (bugs, regressions). Depending on what happened, we may choose to fix it downstream.

The main reasons why we would do so are:
 1. The upstream code is more correct, and we were abusing of the previous code.
 2. The upstream code is more restrictive, and we can add a local change to overcome the restrictions on our subset of IR patterns.
 3. The fix is a long string of changes upstream that won't take place soon, and there is a work around that we can do locally.

On cases 1 and 2, the right place to fix is locally. On case 3, the right place is upstream, but we need a work around. In that case, the work around _must_ be upstreamable, and we need to have a concerted effort (including an issue in Github) to upstream those changes.

The rule of thumb is: if it's a problem with our code, we fix it downstream, if it's a problem with LLVM, we fix it upstream. But the latter may take some time, so we allow for work-arounds until we fix them upstream, but it's not stable to keep work-arounds for a long time.

### Fixing Upstream Bugs Upstream

Contributing to upstream projects can be daunting, but once you get the hang of it it's actually pretty trivial. You just need to make sure you go throught the right hoops, and each community has slightly different hoops, but all of them have a basic set:

1. **You have to convince the change is worth having:** Some changes are obvious (a bug fix with a test case), others not so much (a new feature or a different implementation). For the latter, you may have to do a Request-For-Comment (RFC) on some mailing list or discussion forum to reach a consensus first, then adapt your code to match and post the review.
2. **You have to show that your change is the best way to do it:** It might not be the best long term, but it should be a good direction towards the best overall goal. This requires explanation of why you chose this and as important, why you haven't chosen other ways. You may get stuck in a loop if you don't do that, so it's always good to know your options and why you haven't chosen them.
3. **Your code has to follow the project's policies:** Code format, comment quality, usage of language features all have to match the original source, as well as API design and usage, especially if the code interacts with other parts of the compiler.
4. **Your tests clearly demonstrate the effects on upstream code:** Your problem may be downstream, but your tests need to show the problem with upstream examples. The start of an upstream patch is usually building such an example, that will become the driver for your changes and the final test to prove completeness.

Basically, those rules are steps to show you have done your homework and have shown your work.
LLVM has its own rules for upstreaming in the developer policy, read it before trying to upstream and save yourself some time and avoid upsetting upstream reviewers.

Once the fixes are upstream (and stable, passing buildbots and not reverted), you can pull them through an LLVM version bump. If your changes are reverted soon after, you must try again until it's accepted for good in tree, so that we don't need to cherry-pick anything. Having a good test also makes sure we won't have to fix it again later, if some edge case shows up later.