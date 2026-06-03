"""Feasibility-region analysis (Path B): analytic max-suits M* per scenario.

M/D/1 on a shared slice: latency L(rho) = rho*S/(2(1-rho)) + S + prop.
Budget L <= L_b  <=>  rho <= rho*, with
    rho*/(1-rho*) = 2(L_b - S - prop)/S.
Since aggregate rho = M * r / C  (r = per-suit rate, C = capacity in msg/s),
the max number of suits within budget is the FEASIBILITY BOUND:
    M*(r) = rho* * C / r.
For S0/S1, r is constant -> closed-form M*.
For S2, r=r(M) decreases with congestion -> M* is the fixed point r(M*)=rho*C/M*.
"""
import json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PAYLOAD_B = 200
BW = 12.8e6
S = PAYLOAD_B * 8 / BW * 1000.0     # ms
CAP = BW / (PAYLOAD_B * 8)          # msg/s
PROP = 0.1
BUDGET = 1.0

Wstar = BUDGET - S - PROP
rho_star = (2 * Wstar / S) / (1 + 2 * Wstar / S)
print(f"S={S:.4f} ms  C={CAP:.0f} msg/s  prop={PROP} ms  budget={BUDGET} ms")
print(f"rho* = {rho_star:.4f}   ->  M*(r) = rho* * C / r = {rho_star*CAP:.0f} / r\n")

def Mstar(r):
    return rho_star * CAP / r

# S0/S1 constant rates from the sweep
rows = json.load(open(os.path.expanduser("results/logs/pathB_sweep.json")))
r_s0 = next(x["per_suit_rate"] for x in rows if x["scenario"] == "s0")
r_s1 = next(x["per_suit_rate"] for x in rows if x["scenario"] == "s1")
print(f"S0: r={r_s0:.1f} msg/s -> M* = {Mstar(r_s0):.2f}  => {int(np.floor(Mstar(r_s0)))} suits")
print(f"S1: r={r_s1:.1f} msg/s -> M* = {Mstar(r_s1):.2f}  => {int(np.floor(Mstar(r_s1)))} suits")

# S2: fixed point from r(M) table
Ms = sorted({x["M"] for x in rows})
r_s2 = np.array([next(x["per_suit_rate"] for x in rows if x["M"]==M and x["scenario"]=="s2") for M in Ms])
iso = rho_star * CAP / np.array(Ms, float)   # feasibility curve r*(M)
feasible = r_s2 <= iso
print("\nS2 per-suit rate vs feasibility curve r*(M)=rho*C/M:")
for M, r, rs, ok in zip(Ms, r_s2, iso, feasible):
    print(f"  M={M:3d}: r_s2={r:6.1f}  r*={rs:6.1f}  {'feasible' if ok else 'OVER'}")
print("  (S2 ceiling is beyond M=50 — extend sweep to locate exact M*.)")

# Figure B4: feasibility plane
fig, ax = plt.subplots(figsize=(7, 4.8))
Mgrid = np.linspace(1, 70, 400)
ax.plot(Mgrid, rho_star*CAP/Mgrid, "k-", lw=2, label=r"Feasibility bound  $r^*(M)=\rho^* C/M$")
ax.fill_between(Mgrid, 0, rho_star*CAP/Mgrid, color="green", alpha=0.08)
ax.axhline(r_s0, color="#d62728", ls="-",  lw=2, label=f"S0 r={r_s0:.0f}")
ax.axhline(r_s1, color="#ff7f0e", ls="--", lw=2, label=f"S1 r={r_s1:.0f}")
ax.plot(Ms, r_s2, "o-", color="#1f77b4", lw=2, label="S2 r(M) (adaptive)")
for r, M, c in [(r_s0, Mstar(r_s0), "#d62728"), (r_s1, Mstar(r_s1), "#ff7f0e")]:
    ax.plot(M, r, marker="*", color=c, ms=16, mec="k", zorder=5)
ax.set_xlabel("Number of concurrent suits  M")
ax.set_ylabel("Per-suit message rate r (msg/s)")
ax.set_title("Feasibility Plane: where each policy's rate meets the budget bound\n"
             "(below the black curve = within 1 ms; ★ = S0/S1 max suits)")
ax.set_yscale("log"); ax.set_xlim(0, 70); ax.legend(fontsize=9)
fig.tight_layout()
out = os.path.expanduser("results/figures/figB4_feasibility.png")
fig.savefig(out, bbox_inches="tight"); print(f"\nSaved {out}")
