# Contributing

## Scope

This repository focuses on methodology-oriented simulation code for LLM evaluation governance in LIS contexts. Contributions should preserve reproducibility, readable reporting, and minimal operational complexity.

## How To Contribute

1. Open an issue before making large structural or methodological changes.
2. Keep pull requests focused on one concern at a time.
3. Add or update smoke tests when changing entrypoints, packaging, or workflow wiring.
4. Update documentation when behavior, outputs, or commands change.

## Development Checklist

Before opening a pull request:

1. Run `python -m unittest discover -s tests -p "test_*.py" -v`.
2. Verify `python main.py --help` still works.
3. Keep generated files under `results/` out of commits.

## Style

- Follow existing module structure under `src/`.
- Prefer small, explicit changes over broad rewrites.
- Keep public-facing text in README and docs concise and reproducible.
