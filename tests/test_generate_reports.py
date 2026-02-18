"""Tests for generate_reports CLI (mock data, no DB)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest


def _mock_ablation_json(tmp_path):
    data = {
        "results": [
            {
                "variant": v,
                "algorithm": "ppo",
                "seed": s,
                "sharpe": 0.5,
                "sortino": 0.6,
                "calmar": 0.4,
                "max_drawdown": 0.05,
                "hit_rate": 52.0,
                "turnover": 1.0,
                "regime_metrics_test": {
                    "low": {"annualized_sharpe": 0.3, "sortino": 0.4, "calmar": 0.2, "max_drawdown": 0.03, "hit_rate_pct": 50.0, "turnover_rate": 0.8, "n_bars": 50},
                    "medium": {"annualized_sharpe": 0.5, "sortino": 0.6, "calmar": 0.4, "max_drawdown": 0.05, "hit_rate_pct": 52.0, "turnover_rate": 1.0, "n_bars": 30},
                    "high": {"annualized_sharpe": 0.4, "sortino": 0.5, "calmar": 0.3, "max_drawdown": 0.06, "hit_rate_pct": 48.0, "turnover_rate": 1.2, "n_bars": 20},
                },
            }
            for v in ("A", "B", "C", "D") for s in range(2)
        ],
        "aggregated": [
            {"variant": v, "algorithm": "ppo", "sharpe_mean": 0.5, "sharpe_std": 0.05, "sortino_mean": 0.6, "sortino_std": 0.06, "calmar_mean": 0.4, "calmar_std": 0.04, "max_drawdown_mean": 0.05, "max_drawdown_std": 0.005, "hit_rate_mean": 52.0, "hit_rate_std": 2.0, "turnover_mean": 1.0, "turnover_std": 0.1}
            for v in ("A", "B", "C", "D")
        ],
        "pvalues_vs_A": {"pval_sharpe_vs_A_B_ppo": 0.02, "pval_sharpe_vs_A_C_ppo": 0.05, "pval_sharpe_vs_A_D_ppo": 0.01},
        "config": {},
    }
    path = tmp_path / "ablation.json"
    with open(path, "w") as f:
        json.dump(data, f)
    return path


def test_generate_reports_tables_and_plots(tmp_path):
    """Run generate_reports with mock ablation JSON; no DB so fake bars, no models."""
    sys_path = __import__("sys").path
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys_path:
        sys_path.insert(0, str(root))

    ablation_path = _mock_ablation_json(tmp_path)
    out_dir = tmp_path / "reports"

    from scripts.generate_reports import main
    main(
        ablation_json=ablation_path,
        output_dir=out_dir,
        feature_bars_db=None,
        models_dir=tmp_path / "nonexistent_models",
        limit_bars=80,
    )

    assert (out_dir / "ablation_table_ppo.csv").exists()
    assert (out_dir / "ablation_table_ppo.md").exists()
    assert (out_dir / "regime_table_ppo.csv").exists()
    assert (out_dir / "regime_table_ppo.md").exists()
    assert (out_dir / "equity_curves.png").exists()
    assert (out_dir / "drawdown_over_time.png").exists()
    assert (out_dir / "rolling_sharpe_30bar.png").exists()
