# llm_eval_simulation

Methodology-oriented simulation framework for multi-model LLM evaluation governance in Library and Information Science (LIS).

## Overview

This repository implements a four-layer decision pipeline for situations where many LLM evaluators score the same collection candidates and the goal is to turn those scores into auditable governance decisions.

The framework is designed as a decision-governance workflow rather than a predictive classifier. It separates evaluator reliability, group consensus, decision partitioning, and optional skill diagnosis into explicit stages.

## Core Workflow

1. Layer 1: reliability screening with ICC and confidence-interval-aware boundary retention.
2. Layer 2: consensus subset search with Kendall's W, greedy disagreement removal, and MDS diagnostics.
3. Layer 3: nine-quadrant decision matrix using median score plus rank IQR majority-consensus dispersion.
4. Layer 4: optional skill-level diagnosis for disputed books.

## Current Scale

- 10,000 simulated books
- 100 evaluators grouped into 10 stress-test cohorts
- 2 independent scoring rounds
- 5 LIS-oriented diagnosis dimensions

## Repository Layout

```text
.
|-- main.py                     # Thin root entrypoint
|-- requirements.txt           # Runtime dependencies
|-- scripts/                   # Utility scripts such as DOCX generation
|-- src/                       # Core package modules and pipeline orchestration
|-- docs/                      # Structure and GitHub publishing documentation
`-- results/                   # Generated artifacts (ignored by git)
```

Additional structure notes are available in `docs/PROJECT_STRUCTURE.md`.

## Quick Start

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the main pipeline:

```bash
python main.py
```

Run the main pipeline with Monte Carlo robustness analysis:

```bash
python main.py --monte-carlo
```

## Other Entry Points

Small-scale scenarios:

```bash
python -m src.small_scale_experiment
```

ICC(2,1) vs ICC(3,1) comparison:

```bash
python -m src.icc_comparison_experiment
```

Alternative Layer-3 method benchmark:

```bash
python -c "from src.alternative_methods import run_alternative_methods_report; run_alternative_methods_report()"
```

Generate manuscript draft DOCX:

```bash
python scripts/generate_manuscript_docx.py
```

## Main Outputs

Generated artifacts are written under `results/`, including:

- `results/reports/summary_report.txt`
- `results/reports/bootstrap_report.txt`
- `results/reports/sensitivity_report.txt`
- `results/reports/monte_carlo_report.txt`
- `results/reports/icc_comparison_report.txt`
- `results/reports/alternative_methods_report.txt`

The `results/` directory is intentionally ignored by git so the public repository stays lightweight and reproducible.

## Engineering Notes

- The root entrypoint now delegates to `src.pipeline`, which makes the project easier to test, package, and reuse.
- Output directories are resolved from the project root instead of the current working directory, which prevents misplaced artifacts when running from another shell location.
- CI configuration is included for lightweight install and import checks on GitHub Actions.
- A minimal smoke test suite is included under `tests/` for fast repository sanity checks.

## License

This project is released under the MIT License. See the `LICENSE` file for details.

## Validity Caveats

1. Reliability is not validity.
2. Greedy pruning can exclude coherent minority schools.
3. Quantile partitions are distribution-relative.
4. Density and percentile methods encode different governance viewpoints.
5. Synthetic stress tests still require external validation on real datasets.

## Publishing

The repository already has an `origin` remote configured. For a clean public upload workflow, see `docs/GITHUB_PUBLISH.md`.
