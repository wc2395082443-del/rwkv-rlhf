# Contributing to Helicopter

Helicopter follows a pull-request based review flow for changes to `main`.

## Pull request requirements

- Open a pull request against `main`.
- Fill in the PR template with the purpose, test plan, and test results.
- Keep the branch current with `main` before merge.
- Wait for all required checks to pass.
- Get at least one approval from a maintainer or code owner.
- Resolve all review conversations before merge.

## DCO and sign-off

Contributions must follow the Developer Certificate of Origin in `DCO`.
Every commit in a pull request must include a `Signed-off-by:` trailer:

```text
Signed-off-by: Your Name <your.email@example.com>
```

Use `git commit -s` to add the trailer automatically:

```bash
git commit -s -m "type(scope): summary"
```

To add sign-off trailers to recent commits, use an interactive rebase or:

```bash
git rebase --signoff main
```

## Lightweight local checks

The root CLI tests use the Python standard library and do not need the full
RWKV dependency group:

```bash
PYTHONPATH=src/cli python -m unittest discover -s tests
```
