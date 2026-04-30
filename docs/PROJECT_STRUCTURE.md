# Project Structure

## Design Goals

This repository is organized to keep methodology code, execution entrypoints, generated artifacts, and publication utilities clearly separated.

## Top-Level Folders

- `src/`: core package modules, including the main pipeline orchestrator and each analysis layer.
- `scripts/`: auxiliary utilities that are not part of the main runtime package, such as manuscript export.
- `docs/`: repository-facing documentation for contributors and GitHub readers.
- `results/`: generated artifacts and reports. This folder is ignored by git.

## Important Files

- `main.py`: thin compatibility entrypoint for root-level execution.
- `src/pipeline.py`: central orchestration logic for the full workflow.
- `src/config.py`: project configuration and project-root-aware output paths.
- `requirements.txt`: runtime dependencies for local execution.
- `pyproject.toml`: package metadata and console-script definition.

## Why This Layout

The main optimization in this cleanup is separating orchestration from the root script. That makes the project easier to:

1. run from different working directories,
2. package and expose as a CLI,
3. validate in CI,
4. document for public GitHub consumption.

## Suggested Future Additions

If the repository keeps growing, the next clean split would be:

1. `tests/` for smoke and regression coverage,
2. `notebooks/` for exploratory analysis notebooks,
3. `data/` or `artifacts/` only if reproducible non-generated inputs are added.