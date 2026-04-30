# llm_eval_simulation

A simulation project for methodology research on multi-model LLM evaluation governance in Library and Information Science (LIS).

## Project Scope And Status

- Scope: simulation experiments for a decision-matrix-oriented evaluation governance paper.
- Progress: core experiments are largely complete.
- Current focus: manuscript writing and consolidation of publication-ready evidence.

## What This Project Solves

The project implements a four-layer decision framework to:

1. screen reliable evaluators,
2. aggregate a high-consensus evaluator subset,
3. produce actionable nine-quadrant decisions,
4. optionally diagnose disputed items by skill dimensions.

The framework is positioned as a **decision governance tool** rather than an optimal classifier.

## Four-Layer Framework

1. **Layer 1 (Reliability)**
	- ICC(2,1)-based evaluator screening with CI-assisted boundary handling.
2. **Layer 2 (Consensus)**
	- Kendall's W thresholding (`theta_con=0.80`) with greedy disagreement removal and MDS diagnostics.
3. **Layer 3 (Decision Matrix)**
	- Median score + rank IQR (`Q3-Q1`) majority-consensus dispersion.
	- Default percentile partitioning is `20/40/40`.
4. **Layer 4 (Optional Diagnosis)**
	- Skill-level diagnosis for disputed books:
	  - Scholarly Value
	  - Topical Relevance
	  - Readability
	  - Authority/Credibility
	  - Collection Fit

## Experiment Scale

- `10,000` books
- `100` evaluators in `10` stress-test groups (A-J)
- `2` independent scoring rounds

## Key Current Results

- Layer 1 retained `90` evaluators.
- Layer 2 retained `85` evaluators at `W=0.801`.
- Layer 3 (default percentile + IQR) produced `1,887` high-score/high-consensus recommendations.
- Bootstrap (1,000 rounds) shows high partition stability (`20/40/40` mean stability `0.9858`).
- Monte Carlo (10 seeds) shows low volatility (final `W` mean `0.8065`, SD `0.0031`).

## Installation

```bash
pip install -r requirements.txt
```

## Main Run

```bash
python main.py
```

## Small-Scale Scenarios (3 / 6 / 9 Models)

Run the practical small-scale experiments with 10,000 books for each scenario:

```bash
python -m src.small_scale_experiment
```

Outputs are written to:

- `results/small_scale/small_scale_scenario1_report.txt`
- `results/small_scale/small_scale_scenario2_report.txt`
- `results/small_scale/small_scale_scenario3_report.txt`
- `results/small_scale/small_scale_scenario1_decision_matrix.csv`
- `results/small_scale/small_scale_scenario2_decision_matrix.csv`
- `results/small_scale/small_scale_scenario3_decision_matrix.csv`
- `results/small_scale/small_scale_scenario1_decision_scatter.png`
- `results/small_scale/small_scale_scenario2_decision_scatter.png`
- `results/small_scale/small_scale_scenario3_decision_scatter.png`
- `results/reports/small_scale_comparison_report.txt`

## ICC(2,1) vs ICC(3,1) Comparison Experiment

Run one-click ICC comparison on the same simulated dataset (seed=42):

```bash
python -m src.icc_comparison_experiment
```

Key output:

- `results/reports/icc_comparison_report.txt`
- `results/icc_comparison/model_icc_comparison.csv`
- `results/icc_comparison/layer1_pass_mismatch.csv`
- `results/icc_comparison/layer23_metrics_comparison.csv`

Outputs are written under `results/` with structured subfolders:

- `results/layer1`
- `results/layer2`
- `results/layer3`
- `results/layer4`
- `results/sensitivity`
- `results/bootstrap`
- `results/monte_carlo`
- `results/reports`

## Alternative Layer-3 Methods Benchmark

```bash
python -c "from src.alternative_methods import run_alternative_methods_report; run_alternative_methods_report()"
```

This writes:

- `results/reports/alternative_methods_report.txt`
- charts under `results/alternative/`

## Generate SSCI Manuscript Draft (DOCX)

```bash
python scripts/generate_manuscript_docx.py
```

Output file:

- `results/reports/manuscript_draft.docx`

## Important Reports

- `results/reports/summary_report.txt`
- `results/reports/bootstrap_report.txt`
- `results/reports/sensitivity_report.txt`
- `results/reports/monte_carlo_report.txt`
- `results/reports/final_assessment_report.txt`
- `results/reports/alternative_methods_report.txt`

## Notes on Validity

Known caveats documented in reports and manuscript draft:

1. Reliability is not validity.
2. Greedy pruning may remove coherent minority schools.
3. Quantile boundaries are distribution-relative.
4. Density and percentile methods encode different decision viewpoints.
5. Synthetic stress tests still require external validation.
