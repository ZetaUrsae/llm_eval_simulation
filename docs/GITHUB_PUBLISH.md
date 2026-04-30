# GitHub Publish Guide

## Current State

This repository already has an `origin` remote configured:

```text
https://github.com/ZetaUrsae/llm_eval_simulation.git
```

## Before Pushing Publicly

1. Confirm that `results/` remains ignored so generated artifacts do not bloat the repository.
2. Review `ReadMe.md` and metadata in `pyproject.toml`.
3. Decide whether to add a license file before making the repository public.
4. Check that no local-only data, credentials, or temporary notes are present.

## Recommended Commands

From the repository root:

```bash
git status
git add .
git commit -m "Refactor project structure and prepare public repo"
git push origin main
```

If the default branch is `master`, replace the final command with:

```bash
git push origin master
```

## Make The Repository Public

On GitHub:

1. Open the repository settings.
2. Go to `General`.
3. Scroll to `Danger Zone`.
4. Use `Change repository visibility` and select `Public`.

## Optional Next Steps

1. Add a `LICENSE` file.
2. Create the first release tag after the public push.
3. Enable branch protection and required CI checks.
4. Add screenshots or sample report snippets to the README.