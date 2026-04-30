# llm_eval_simulation

Research codebase for simulation-oriented study of multi-model LLM evaluation workflows in Library and Information Science (LIS).

## Status

This repository supports ongoing academic work. Public materials are intentionally concise during submission and review cycles.

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

Optional robustness run:

```bash
python main.py --monte-carlo
```

## Additional Commands

```bash
python -m src.small_scale_experiment
python -m src.icc_comparison_experiment
python -c "from src.alternative_methods import run_alternative_methods_report; run_alternative_methods_report()"
python scripts/generate_manuscript_docx.py
```

## Outputs

Run artifacts are generated under `results/` and are not tracked in git.

## Reproducibility Note

Implementation and documentation may be updated alongside peer-review revisions.

## License

MIT License. See `LICENSE`.

## Community

Contribution, conduct, security, and template files are included for public collaboration.
