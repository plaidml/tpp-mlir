# How to contribute to TPP MLIR

# Open an issue

If you found a bug, wants some feature implemented or just want to discuss implementation details, just open an [issue](https://github.com/plaidml/tpp-mlir/issues).

In this case:
 * Write a summary of the problem, with as much information as you can (error messages, code, etc)
 * Clearly state what you want from it (expected behaviour, problem resolution, etc)
 * If you don't plan working on it, leave the issue unassigned
 * The core developers are on LLVM's IRC, discord and discourse, you can ping them there

Core developers can use issues as a way to track their progress, and even mark dependency between tasks to plan a development strategy.

In this case:
 * Write a summary of the task, what's the end goal, labels if you want
 * Mark as dependency with "depends on #NN"
 * Assign it to yourself, submit PRs referring to the issue number on the commit message
 * On the last PR, just write "Fixes #NN" on the commit message and the issue will be automatically closed

_Note: Don't use "Fixes #NN" on interim PRs as that'll close the issue. Just referring to the issue number directly should be fine._

# Submit a PR

Everyone, including core developers, should work on their own fork.

To submit a [PR](https://github.com/plaidml/tpp-mlir/pulls), create a branch on your fork, push to your origin and submit a PR from that fork/branch. Avoid using your `main` branch, or you'll only be able to contribute with one change at a time.

Standard Git process apply:
 * Work locally on your fork, creating at least one branch with the code you want pulled
 * When ready, create a pull request, referencing an issue (if any)
 * This may trigger a CI loop
   * If it doesn't, ask a core developer to trigger it
   * This situation will improve with [Github Actions](https://github.com/plaidml/tpp-mlir/issues/57), where all PRs will be tested
 * Wait for the CI to be green, fixing any issues to get there
 * Wait for code review, and approval
   * Once you implement the changes requested, mark conversations as _"resolved"_ to indicate completion
   * Avoid force-pushing (`commit --amend` + `push +branch`), just add a new commit with a title/desc that describes what changed
 * Merge the PR once you get approval and CI pass
   * If there's more than one commit, consider `squash & merge` instead of `rebase & merge`
   * If you don't have permission, ask for it to be merged (so we're sure you've finished working on it)

_Note: Core developers may push directly to `main` if the change is deemed non-functional. They will receive a metaphorical "slap on the wrist" if it breaks the build._

# Integration with other tools

If other tools use our code, they should have a fork of this repository, where they keep any delta to make their code work.

Core developers of this repository should pull or cherry-pick code from those repositories if/when needed, create a PR and follow the CI process.

It's up to the developers of the other tools and the core developers of tpp-mlir interested in the integration to synchronise.

This is required to make sure every change is tested with our local CI, as well as other external ones.

# Branches

Since every developer works on their own forks, there is no need for development branches in tree.

This is particularly important because the CI tests every branch of the main repo and it would be testing work-in-progress of everyone for no reason.

However, some exceptions may merit a branch:
 * **Feature branches**: When two or more developers are working on a large feature
   * These should be short lived and not astray from the `main` branch too much
   * Using PRs or even direct pushes (from core developers) to this branch should be fine
   * CI can happen independently, as we do pick branch changes too
   * Do not allow too many LLVM versions to change between this branch and `main`
   * Consider a more fine-grained roadmap approach
 * **Experimental branches**: When some experiment needs to pause and isn't finished yet
   * These could be long lived and we don't care about CI or staleness very much
   * Keeping in the main repo is a declaration of intent that we want others to resume work
   * Someone else can restart work on their own repo without having to fork yours
   * Whoever restarts work is responsible for rebasing and updating LLVM, etc
   * This should be mostly infrastructure related (CMake, CI, etc)
 * **CI branches**: When you want to run CI on some changes but are not quite ready for a PR
   * Consider opening a Draft PR, instead

Please, make sure to delete any branch in the main repo that is stale or no longer needed.
