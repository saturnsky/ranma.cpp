# Branches, tags and releases

RANMA.cpp is a series of changes on top of upstream llama.cpp. The series is rebuilt for
every release instead of growing by merges, so that every change keeps a clear place and
can be submitted upstream or dropped on its own.

## Refs

| ref | kind | what it points to | changes |
|---|---|---|---|
| `master` | branch | the upstream llama.cpp `master` commit that the latest release (`ranma`) is based on | at every release, to the base of that release |
| `ranma_YYYYMMDD` | tag; also a branch for the three latest releases | a release: the series as it was published on that date, with its benchmark | never |
| `ranma` | branch | the latest release (the same commit as the newest `ranma_YYYYMMDD` tag) | at every release |
| `ranma_upstream` | branch | the latest release rebased onto a newer upstream commit, without new work; the starting point of the next release | after every release and whenever it is rebased again |
| `features/*` | branch | changes prepared for upstream pull requests | as the upstream review needs |

`master` is a plain copy of upstream; it carries no change of this fork. `git diff master ranma`
shows exactly what the latest release adds to its upstream base. `ranma_upstream` is usually
based on a newer upstream commit than `master`.

The three latest releases are also kept as branches of the same name, pointing at the same commit as
their tag; older releases are tags only. Like the tags, these branches never move.

## Verification level

- **Release tags (`ranma_YYYYMMDD`) and `ranma`**: built and tested on the target environment
  (Windows 11, Radeon gfx1201, HIP). The [benchmarks](benchmarks/README.md) are kept as records, each naming
  the release it was measured at; not every number is measured again for every release.
- **`ranma_upstream`**: build-checked only. Conflicts with the newer upstream are resolved,
  but no benchmark is run. Use a release tag when you need measured behavior.

## Release cycle

1. **Rebase.** After a release, the series is rebased onto the current upstream `master`.
   The result is published right away as `ranma_upstream`. `master` stays at the base of the
   release.
2. **Develop.** New work for the next release starts from `ranma_upstream`.
3. **Rebuild the series.** Before a release every commit is placed where it belongs in the
   series. Changes that belong together are squashed into one commit, follow-up fixes are
   folded into the commit they fix, and changes found to cause a regression are dropped.
   Commit hashes therefore change from release to release.
   - Commits taken unchanged from upstream pull requests that are not merged upstream yet
     (for example a model's draft head) stay byte-for-byte as in the pull request, so that
     they can be replaced by the upstream commit once it is merged.
   - The benchmark of the release is always the last commit of the series.
4. **Publish.** The series is tagged `ranma_YYYYMMDD` (annotated), `ranma` is moved to it, and
   `master` is moved to its upstream base commit. The three latest releases are also kept as
   branches of the same name; older ones are tags only.

## Contributions

Pull requests go to `ranma_upstream`; see [CONTRIBUTING.md](../../CONTRIBUTING.md) for what
happens to an accepted change in step 3.
