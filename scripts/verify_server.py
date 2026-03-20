"""Quick verification script to run on the Hetzner server."""
import numpy as np
from src.envs.options_env import OptionsEnv
from src.envs.reward import compute_reward, check_drawdown_terminate
from src.envs.execution_sim import ExecutionSimulator
from src.agents.train_sb3 import train_agent
from src.agents.ablation import run_ablation

# 1. Reward function
r, peak, _ = compute_reward(
    pnl_step=23.0, transaction_costs=3.25,
    peak_before=100000.0, equity_before=100000.0, equity_after=100023.0,
    lambda_cost=1.0, lambda_dd=0.1,
    has_positions=True, inaction_penalty=0.02, exposure_bonus=0.01,
)
print(f"Reward with PnL=+$23, fee=$3.25, holding: {r:.4f}")
assert r > 0, f"Expected positive reward, got {r}"

r2, _, _ = compute_reward(
    pnl_step=0.0, transaction_costs=0.0,
    peak_before=100000.0, equity_before=100000.0, equity_after=100000.0,
    lambda_cost=1.0, lambda_dd=0.1,
    has_positions=False, inaction_penalty=0.02, exposure_bonus=0.01,
)
print(f"Reward doing nothing: {r2:.4f}")
assert r2 == -0.02, f"Expected -0.02, got {r2}"

# 2. Drawdown termination
assert check_drawdown_terminate(100000, 85000, 0.15) == True
assert check_drawdown_terminate(100000, 86000, 0.15) == False
print("Drawdown termination checks: PASS")

# 3. Env smoke test
bars = []
for i in range(100):
    bars.append({
        "underlying": "SPY", "atm_iv_7d": 0.18, "atm_iv_14d": 0.19,
        "atm_iv_30d": 0.20, "iv_term_slope": 0.01, "iv_skew": -0.05,
        "realized_vol_5d": 0.15, "realized_vol_10d": 0.16,
        "realized_vol_20d": 0.17, "realized_vol_60d": 0.18,
        "vix_close": 18.0, "equity_return_1d": 0.001 * ((-1) ** i),
        "sent_news_asset": 0.1, "sent_social_asset": 0.05,
        "sent_macro_topic": -0.1, "sent_dispersion": 0.2,
        "sent_momentum": 0.0, "sent_volume": 50.0,
        "no_news_flag": False, "options_gap_flag": False,
    })
env = OptionsEnv(feature_bars=bars)
obs, info = env.reset()
assert obs.shape == (52,), f"Bad obs shape: {obs.shape}"

# Open straddle
obs2, rew, term, trunc, info2 = env.step([2, 1, 2, 1])
print(f"Step 1 (open straddle): reward={rew:.4f}, equity={info2['equity']:.2f}")

# Hold same (should be near no-op - no new orders needed)
obs3, rew2, _, _, info3 = env.step([2, 1, 2, 1])
print(f"Step 2 (hold same): reward={rew2:.4f}, equity={info3['equity']:.2f}")

# Flatten
obs4, rew3, _, _, info4 = env.step([1, 1, 1, 1])
print(f"Step 3 (flatten): reward={rew3:.4f}, equity={info4['equity']:.2f}")

print()
print("=" * 50)
print("ALL CHECKS PASSED - Ready to run ablation")
print("=" * 50)
