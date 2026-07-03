# Helicopter Remote Agent Notes Example

## GitHub Submission Rules

- Before any commit or push, inspect the parent repository and submodules with
  `git status --short --branch --ignore-submodules=none` and
  `git submodule status --recursive`.
- Treat an uppercase `M` on a submodule path as a parent-repo gitlink change,
  and a lowercase `m` as dirty work inside the submodule. Inspect each changed
  submodule with `git -C <path> status --short --branch` before staging the
  parent repository.
- Do not commit or push the parent repository while any included submodule
  appears as `-dirty`. A parent commit must reference a clean submodule commit
  that has already been committed and pushed to that submodule's configured
  remote.
- If a task changes submodule code, commit and push the submodule first. Then
  update the parent repository gitlink, stage only the intended parent files
  and submodule path, inspect `git diff --cached --submodule=short`, and commit
  the parent repository after the submodule commit is fetchable.
- If a dirty submodule is unrelated to the requested change, leave the
  submodule unstaged, do not advance the parent gitlink, and report the dirty
  submodule as pre-existing or out of scope.
- Use path-limited staging and commits in this repository. Avoid broad
  `git add -A` or repository-wide commits when submodule state is present.
- Final publish reports must list the pushed parent branch and, when applicable,
  each pushed submodule path, branch, and commit hash.
