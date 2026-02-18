"""Tests for ablation runner logic (without full SB3 training)."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest

from src.agents.ablation import (
    SB3PolicyAdapter,
    run_ablation,
    save_ablation_results,
    _empty_run_result,
    VARIANTS,
)
from src.agents.obs_mask_wrapper import VARIANT_MASKS
from src.agents.train_sb3 import split_bars_by_time
from src.envs.options_env import OptionsEnv


def _make_fake_bars(n: int = 80) -> list[dict]:
    return [
        {
            "underlying": "SPY",
            "equity_return_1d": 0.001,
            "realized_vol_5d": 0.15,
            "realized_vol_10d": 0.16,
            "realized_vol_20d": 0.17,
            "realized_vol_60d": 0.18,
            "vix_close": 18.0,
            "iv_term_slope": -0.01,
            "iv_skew": 0.02,
            "options_gap_flag": False,
            "atm_iv_7d": 0.20,
            "atm_iv_14d": 0.19,
            "atm_iv_30d": 0.18,
            "sent_news_asset": 0.1,
            "sent_social_asset": 0.0,
            "sent_macro_topic": -0.05,
            "sent_dispersion": 0.02,
            "sent_momentum": 0.01,
            "sent_volume": 5,
            "no_news_flag": False,
        }
        for _ in range(n)
    ]


def test_split_bars_by_time():
    bars = _make_fake_bars(100)
    train, val, test = split_bars_by_time(bars, train_pct=0.70, val_pct=0.15)
    assert len(train) == 70
    assert len(val) == 15
    assert len(test) == 15
    train2, val2, test2 = split_bars_by_time([], train_pct=0.7, val_pct=0.15)
    assert train2 == val2 == test2 == []


def test_empty_run_result():
    r = _empty_run_result("A", "ppo", 0)
    assert r["variant"] == "A" and r["algorithm"] == "ppo" and r["seed"] == 0
    assert r["sharpe"] == 0.0 and r["hit_rate"] == 0.0


def test_save_ablation_results_structure():
    out = {
        "results": [
            {"variant": "A", "algorithm": "ppo", "seed": 0, "sharpe": 0.1, "sortino": 0.1, "calmar": 0.0, "max_drawdown": 0.0, "hit_rate": 50.0, "turnover": 0.0},
        ],
        "aggregated": [
            {"variant": "A", "algorithm": "ppo", "sharpe_mean": 0.1, "sharpe_std": 0.0, "sortino_mean": 0.1, "sortino_std": 0.0, "calmar_mean": 0.0, "calmar_std": 0.0, "max_drawdown_mean": 0.0, "max_drawdown_std": 0.0, "hit_rate_mean": 50.0, "hit_rate_std": 0.0, "turnover_mean": 0.0, "turnover_std": 0.0},
        ],
        "pvalues_vs_A": {},
        "config": {"algorithms": ["ppo"], "seeds": [0], "total_timesteps": 100, "train_pct": 0.7, "val_pct": 0.15},
    }
    with tempfile.TemporaryDirectory() as d:
        jp = Path(d) / "out.json"
        cp = Path(d) / "out.csv"
        save_ablation_results(out, json_path=jp, csv_path=cp)
        assert jp.exists() and cp.exists()
        with open(jp) as f:
            loaded = json.load(f)
        assert "results" in loaded and "aggregated" in loaded
        assert len(open(cp).readlines()) >= 2


@pytest.mark.skipif(
    __import__("importlib").util.find_spec("stable_baselines3") is None,
    reason="stable-baselines3 not installed",
)
def test_run_ablation_single_run():
    """Run ablation with 1 variant, 1 algorithm, 1 seed, few timesteps to test pipeline."""
    bars = _make_fake_bars(120)
    with tempfile.TemporaryDirectory() as tmp:
        models_dir = Path(tmp) / "models"
        out = run_ablation(
            feature_bars=bars,
            algorithms=("ppo",),
            seeds=[0],
            total_timesteps=200,
            models_dir=models_dir,
            train_pct=0.70,
            val_pct=0.15,
        )
    assert "results" in out and "aggregated" in out and "pvalues_vs_A" in out
    assert len(out["results"]) >= 4
    runs_a = [r for r in out["results"] if r["variant"] == "A" and r["algorithm"] == "ppo"]
    assert len(runs_a) == 1
    assert "sharpe" in runs_a[0] and "hit_rate" in runs_a[0]
    assert len(out["aggregated"]) >= 4
