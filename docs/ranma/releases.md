# Branches, tags and releases

RANMA.cpp is a series of changes on top of upstream llama.cpp. The series is rebuilt for
every release instead of growing by merges, so that every change keeps a clear place and
can be submitted upstream or dropped on its own.

## Refs

| ref | kind | what it points to | changes |
|---|---|---|---|
| `master` | branch | the upstream llama.cpp `master` commit that the latest release (`ranma`) is based on | at every release, to the base of that release |
| `ranma_YYYYMMDD` | tag; also a branch for the three latest releases | a release: the series as it was published on that date, with its benchmark | never |
| `ranma` | branch | the latest release: the newest `ranma_YYYYMMDD` tag, plus any later commits that change only the release tooling or documentation, not the code | at every release, and for such tooling or documentation commits |
| `ranma_upstream` | branch | the latest release rebased onto a newer upstream commit, with its overlaps with upstream resolved and without new features; the baseline of further work and the starting point of the next release | after every release and whenever it is rebased again |
| `features/*` | branch | changes prepared for upstream pull requests | as the upstream review needs |

`master` is a plain copy of upstream; it carries no change of this fork. `git diff master ranma_YYYYMMDD`
shows exactly what a release adds to its upstream base. `ranma_upstream` is usually
based on a newer upstream commit than `master`.

The three latest releases are also kept as branches of the same name, pointing at the same commit as
their tag; older releases are tags only. Like the tags, these branches never move.

## Verification level

- **Release tags (`ranma_YYYYMMDD`) and `ranma`**: built and tested on the target environment
  (Windows 11, Radeon gfx1201, HIP). The [benchmarks](benchmarks/README.md) are kept as records, each naming
  the release it was measured at; not every number is measured again for every release.
- **Release binaries** (zip files attached to a GitHub Release): built from the release tag by
  the `RANMA release (Windows HIP)` workflow on a GitHub-hosted runner, for the GPU targets named in
  the file name. Before the release is published, the maintainer runs `test-backend-ops` and a short
  generation with the downloaded zip on gfx1201. The benchmarks were measured with the maintainer's
  own builds, not with these binaries; the zip is built with `GGML_NATIVE=OFF`.
- **`ranma_upstream`**: the baseline that further work builds on. Where the newer upstream now
  does what a change of the series did, the change is dropped, or rebuilt on top of the upstream
  implementation when it extends it, so that no two mechanisms do the same job. The places that
  changed are built and tested on the target environment; the benchmark is not run again. Use a
  release tag when you need measured numbers.

## Release cycle

1. **Rebase.** After a release, the series is rebased onto the current upstream `master`.
   Overlaps with the newer upstream are resolved in the same step (see the verification level
   above), and the result is published as `ranma_upstream`. `master` stays at the base of the
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
   Pushing the tag starts the `RANMA release (Windows HIP)` workflow
   (`.github/workflows/ranma-release.yml`), which builds the Windows HIP binaries and attaches them
   to a draft GitHub Release for that tag. The maintainer downloads the zip, runs the local smoke
   test described under Verification level and then publishes the release. The workflow never
   publishes a release and never changes a published one; it can also be started by hand for an
   existing tag.

## Contributions

Pull requests go to `ranma_upstream`; see [CONTRIBUTING.md](../../CONTRIBUTING.md) for what
happens to an accepted change in step 3.
