"""Minimal smoke tests for package structure and CLI wiring."""

from __future__ import annotations

import unittest

from src import config
from src import pipeline
from src.small_scale_experiment import _build_parser as build_small_scale_parser


class ProjectSmokeTests(unittest.TestCase):
    """Fast smoke checks that do not execute the full simulation pipeline."""

    def test_results_dir_is_project_root_relative(self) -> None:
        self.assertEqual(config.RESULTS_DIR, config.PROJECT_ROOT / "results")

    def test_main_parser_accepts_monte_carlo_flag(self) -> None:
        args = pipeline.build_parser().parse_args(["--monte-carlo"])
        self.assertTrue(args.monte_carlo)

    def test_small_scale_parser_default_seed(self) -> None:
        args = build_small_scale_parser().parse_args([])
        self.assertEqual(args.seed, config.SEED)

    def test_pipeline_entrypoint_is_callable(self) -> None:
        self.assertTrue(callable(pipeline.main))


if __name__ == "__main__":
    unittest.main()