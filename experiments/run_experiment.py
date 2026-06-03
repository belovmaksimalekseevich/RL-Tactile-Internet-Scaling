"""
Experiment runner — PATH B: M telepresence suits share one 6G tactile slice.

For a given number of competing suits M:
  * each suit runs the same sampling policy (mean-field),
  * aggregate slice load = M * per_suit_rate,
  * shared-channel latency from M/D/1 (SimPy-validated).

S2 (RL) observes congestion (M) and adapts; S0/S1 are M-agnostic.

Usage:
  python experiments/run_experiment.py --scenario s0 --M 10
  python experiments/run_experiment.py --scenario s1 --M 10 --theta 2.0
  python experiments/run_experiment.py --scenario s2 --M 10 --model results/checkpoints/ppo_s2_B.zip
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse, json
import numpy as np
from pathlib import Path

from src.loaders.dip_loader     import load_subjects as load_dip
from src.loaders.ninapro_loader import load_subjects as load_ninapro
from src.network.channel        import ChannelModel, ChannelConfig

RESULTS_DIR = Path(__file__).parent.parent / "results" / "logs"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

JND_BODY_DEG = 2.0
JND_HAND_DEG = 3.0
E_TX_uJ         = 0.5
E_IDLE_uJ_per_s = 20.0

# Path-B constants (must match env.py / train.py)
R_PER_SUIT   = 3540.0
M_MAX        = 50
SLICE_BW_BPS = 12.8e6      # shared tactile slice capacity
BUDGET_MS    = 1.0


# ----------------------- send masks -----------------------
def mask_s0(angles):
    return np.ones(angles.shape, dtype=bool)

def mask_s1(angles, theta):
    T, J = angles.shape
    m = np.zeros((T, J), dtype=bool); m[0] = True
    last = angles[0].copy()
    for t in range(1, T):
        d = np.abs(angles[t] - last); s = d > theta
        m[t] = s; last[s] = angles[t][s]
    return m

def mask_s2(angles, policy, channel, fps, M):
    T, J = angles.shape
    m = np.zeros((T, J), dtype=bool); m[0] = True
    last = angles[0].copy(); last_t = np.zeros(J, dtype=int); cnt = J
    congestion = M / float(M_MAX)
    for t in range(1, T):
        p = cnt / (t * J)
        aggregate = M * p * R_PER_SUIT
        lat = channel.mean_latency_ms(aggregate)
        lat_rem = float(np.clip((BUDGET_MS - lat) / BUDGET_MS, -1.0, 1.0))
        cur, prev = angles[t], angles[t - 1]
        prev2 = angles[max(0, t - 2)]
        delta = np.abs(cur - last)
        vel   = np.abs(cur - prev)
        acc   = np.abs(vel - np.abs(prev - prev2))
        tsince = (t - last_t) / fps
        phase  = (vel > 2.0).astype(np.float32)
        obs = np.stack([delta, vel, acc, np.minimum(tsince, 10.0), delta,
                        np.full(J, congestion), np.full(J, lat_rem), phase],
                       axis=1).astype(np.float32)
        a, _ = policy.predict(obs, deterministic=True)
        s = a.astype(bool)
        m[t] = s; last[s] = cur[s]; last_t[s] = t; cnt += int(s.sum())
    return m


# ----------------------- per-stream metrics (no latency) -----------------------
def stream_metrics(angles, mask, fps, jnd):
    T, J = angles.shape
    dur = T / fps
    n_sent = int(mask.sum()); n_total = T * J
    rate = n_sent / max(dur, 1e-9)
    recv = np.zeros_like(angles); last = angles[0].copy()
    for t in range(T):
        last[mask[t]] = angles[t][mask[t]]; recv[t] = last
    rmse = np.sqrt(np.mean((angles - recv) ** 2, axis=1))
    return {
        "n_sent": n_sent, "n_total": n_total,
        "reduction_pct": round(100.0 * (1 - n_sent / n_total), 2),
        "rate_msg_s": round(rate, 1),
        "energy_uJ": round(n_sent * E_TX_uJ + dur * E_IDLE_uJ_per_s, 2),
        "rmse_mean_deg": round(float(rmse.mean()), 4),
        "rmse_max_deg":  round(float(rmse.max()), 4),
        "pct_above_jnd": round(float((rmse > jnd).mean() * 100), 2),
        "jnd_deg": jnd,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=["s0", "s1", "s2"], default="s0")
    ap.add_argument("--dip_subjects",     nargs="+", default=["s_01", "s_02", "s_03"])
    ap.add_argument("--ninapro_subjects", nargs="+", default=["s_1", "s_2", "s_3"])
    ap.add_argument("--M",     type=int,   default=10)
    ap.add_argument("--theta", type=float, default=2.0)
    ap.add_argument("--model", type=str,   default=None)
    ap.add_argument("--tag",   type=str,   default=None)
    args = ap.parse_args()

    channel = ChannelModel(ChannelConfig(bandwidth_bps=SLICE_BW_BPS,
                                         background_load=0.0))
    print(f"\n=== {args.scenario.upper()} | PATH B | M={args.M} suits | "
          f"slice={SLICE_BW_BPS/1e6:.1f} Mbps ===")

    body = np.vstack([s["angles_deg"] for s in load_dip(args.dip_subjects)])
    hand = np.vstack([s["angles_deg"] for s in load_ninapro(args.ninapro_subjects)])

    if args.scenario == "s0":
        bm, hm = mask_s0(body), mask_s0(hand)
    elif args.scenario == "s1":
        bm, hm = mask_s1(body, args.theta), mask_s1(hand, args.theta)
    else:
        from stable_baselines3 import PPO
        policy = PPO.load(args.model)
        print("  body mask..."); bm = mask_s2(body, policy, channel, 60.0,  args.M)
        print("  hand mask..."); hm = mask_s2(hand, policy, channel, 100.0, args.M)

    body_m = stream_metrics(body, bm, 60.0,  JND_BODY_DEG)
    hand_m = stream_metrics(hand, hm, 100.0, JND_HAND_DEG)

    per_suit_rate = body_m["rate_msg_s"] + hand_m["rate_msg_s"]
    aggregate_rate = args.M * per_suit_rate
    cap_msg_s = SLICE_BW_BPS / (channel.cfg.payload_bytes * 8)
    rho = aggregate_rate / cap_msg_s
    lat = channel.mean_latency_ms(aggregate_rate)

    result = {
        "scenario": args.scenario, "M": args.M,
        "slice_mbps": SLICE_BW_BPS / 1e6,
        "body": body_m, "hand": hand_m,
        "per_suit_rate_msg_s": round(per_suit_rate, 1),
        "aggregate": {
            "rate_msg_s": round(aggregate_rate, 1),
            "rho": round(rho, 4),
            "latency_ms": round(lat, 4),
            "within_1ms": bool(lat <= BUDGET_MS and rho < 1.0),
        },
    }

    print(f"  per-suit rate: {per_suit_rate:.0f} msg/s  "
          f"(body red {body_m['reduction_pct']:.1f}%, hand red {hand_m['reduction_pct']:.1f}%)")
    print(f"  body RMSE {body_m['rmse_mean_deg']:.2f} (>{JND_BODY_DEG}: {body_m['pct_above_jnd']:.1f}%)  "
          f"hand RMSE {hand_m['rmse_mean_deg']:.2f} (>{JND_HAND_DEG}: {hand_m['pct_above_jnd']:.1f}%)")
    print(f"  AGGREGATE: {aggregate_rate:.0f} msg/s  rho={rho:.3f}  "
          f"latency={lat:.4f} ms  within_1ms={result['aggregate']['within_1ms']}")

    tag = args.tag if args.tag else f"{args.scenario}B_M{args.M}"
    out = RESULTS_DIR / f"{tag}.json"
    json.dump(result, open(out, "w"), indent=2)
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
