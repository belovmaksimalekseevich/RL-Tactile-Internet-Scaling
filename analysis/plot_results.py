"""Path-B figures from pathB_sweep.json."""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = os.path.expanduser("results/logs/pathB_sweep.json")
OUT = os.path.expanduser("results/figures")
os.makedirs(OUT, exist_ok=True)
rows = json.load(open(LOG))

SC = ["s0", "s1", "s2"]
LAB = {"s0": "S0: Always-On", "s1": "S1: Deadband (θ=2°)", "s2": "S2: PPO-RL (ours)"}
COL = {"s0": "#d62728", "s1": "#ff7f0e", "s2": "#1f77b4"}
MK  = {"s0": "s", "s1": "^", "s2": "o"}
LS  = {"s0": "-", "s1": "--", "s2": "-"}
Ms  = sorted({r["M"] for r in rows})

def series(sc, key, sub=None):
    out = []
    for M in Ms:
        r = next(x for x in rows if x["M"] == M and x["scenario"] == sc)
        out.append(r[sub][key] if sub else r[key])
    return np.array(out, dtype=float)

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                     "legend.fontsize": 10, "figure.dpi": 150,
                     "axes.grid": True, "grid.alpha": 0.3})

# Fig B1: aggregate latency vs M (HEADLINE)
fig, ax = plt.subplots(figsize=(7, 4.8))
for sc in SC:
    lat = series(sc, "latency_ms")
    rho = series(sc, "rho")
    lat_plot = np.where(rho < 1.0, lat, np.nan)  # hide overloaded (diverged) points
    ax.plot(Ms, lat_plot, marker=MK[sc], color=COL[sc], ls=LS[sc],
            lw=2.2, ms=8, label=LAB[sc])
    # mark first overload point
    over = [M for M, r in zip(Ms, rho) if r >= 1.0]
    if over:
        ax.axvline(min(over), color=COL[sc], ls=":", lw=1, alpha=0.5)
ax.axhline(1.0, color="black", ls=":", lw=1.8, label="1 ms tactile budget")
ax.set_xlabel("Number of concurrent suits  M  (shared 12.8 Mbps slice)")
ax.set_ylabel("Aggregate E2E Latency (ms)")
ax.set_title("Shared-Slice Latency vs. Cell Load (Path B)\n"
             "RL adapts per-suit sampling to keep the slice within budget")
ax.set_ylim(0, 2.0)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB1_latency_vs_M.png"), bbox_inches="tight")
plt.close(); print("figB1 saved")

# Fig B2: graceful degradation — per-suit RMSE vs M (S2)
fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(Ms, series("s2", "rmse_mean_deg", "body"), "o-", color="#1f77b4", lw=2, label="Body RMSE (S2)")
ax.plot(Ms, series("s2", "rmse_mean_deg", "hand"), "s-", color="#2ca02c", lw=2, label="Hand RMSE (S2)")
ax.axhline(2.0, color="#1f77b4", ls="--", alpha=0.6, label="Body JND = 2°")
ax.axhline(3.0, color="#2ca02c", ls="--", alpha=0.6, label="Hand JND = 3°")
ax.set_xlabel("Number of concurrent suits  M")
ax.set_ylabel("Per-suit reconstruction RMSE (deg)")
ax.set_title("Graceful Fidelity Degradation under Congestion (S2)\n"
             "Agent trades fidelity for latency only as the cell fills")
ax.legend(loc="upper left"); ax.set_ylim(0, 3.5)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB2_graceful_degradation.png"), bbox_inches="tight")
plt.close(); print("figB2 saved")

# Fig B3: per-suit send rate vs M (adaptivity)
fig, ax = plt.subplots(figsize=(7, 4.5))
for sc in SC:
    ax.plot(Ms, series(sc, "per_suit_rate"), marker=MK[sc], color=COL[sc], ls=LS[sc],
            lw=2, ms=7, label=LAB[sc])
ax.set_xlabel("Number of concurrent suits  M")
ax.set_ylabel("Per-suit message rate (msg/s)")
ax.set_title("Per-suit Sampling Rate vs. Congestion\nS2 reduces its own rate as the cell fills; S0/S1 are blind to load")
ax.legend(); ax.set_yscale("log")
fig.tight_layout(); fig.savefig(os.path.join(OUT, "figB3_rate_vs_M.png"), bbox_inches="tight")
plt.close(); print("figB3 saved")

# Capacity gain summary
print("\nMax M within 1ms budget:")
for sc in SC:
    okM = [r["M"] for r in rows if r["scenario"] == sc and r["within_1ms"]]
    print(f"  {LAB[sc]}: {max(okM) if okM else 0} suits")
