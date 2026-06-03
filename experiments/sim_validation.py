"""
Discrete-event validation of the M/G/1 (M/D/1) analytical channel model.

Goal: confirm that the Pollaczek-Khinchine mean latency used throughout the
paper matches a packet-level discrete-event simulation, AND obtain *true*
p95/p99 tail latencies (the analytical model gives only the mean).

We model the shared 6G uplink as a single FIFO server with deterministic
service time S. Two arrival processes are superposed:
  * background traffic  : Poisson at rate lambda_bg = rho_bg * capacity
  * suit traffic        : either Poisson (to test the M/D/1 assumption) or
                          a bursty/periodic process (to test robustness).

Outputs a JSON with analytical vs simulated mean / p95 / p99 across a sweep.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.expanduser("."))
import numpy as np
import simpy

from src.network.channel import ChannelModel, ChannelConfig

CFG = ChannelConfig()
S_MS        = CFG.payload_bytes * 8 / CFG.bandwidth_bps * 1000.0   # service time, ms
CAP_MSG_S   = CFG.bandwidth_bps / (CFG.payload_bytes * 8)          # capacity, msg/s
PROP_MS     = CFG.prop_delay_ms


def analytical_mean_ms(suit_rate, rho_bg):
    ch = ChannelModel(ChannelConfig(background_load=rho_bg))
    return ch.mean_latency_ms(suit_rate)


def simulate(suit_rate, rho_bg, suit_process="poisson",
             n_packets=200_000, warmup=20_000, seed=0):
    """
    Run a single-server FIFO queue with deterministic service.
    Returns dict of measured sojourn-time stats (ms), incl. propagation.
    """
    rng = np.random.default_rng(seed)
    bg_rate = rho_bg * CAP_MSG_S            # msg/s
    S_s = S_MS / 1000.0                     # service time, s

    # Pre-generate arrival timestamps (seconds) for both streams, merge, sort.
    horizon = (n_packets * 1.5) / max(suit_rate + bg_rate, 1.0)

    # Background: Poisson
    n_bg = int(bg_rate * horizon * 1.2) + 10
    bg_iat = rng.exponential(1.0 / max(bg_rate, 1e-9), n_bg)
    bg_arr = np.cumsum(bg_iat)

    # Suit: Poisson or periodic-with-jitter (bursty)
    n_su = int(suit_rate * horizon * 1.2) + 10
    if suit_process == "poisson":
        su_iat = rng.exponential(1.0 / max(suit_rate, 1e-9), n_su)
    else:  # "periodic": deterministic spacing + small jitter (bursty, D-like)
        base = 1.0 / max(suit_rate, 1e-9)
        su_iat = base * (1.0 + rng.uniform(-0.1, 0.1, n_su))
    su_arr = np.cumsum(su_iat)

    arrivals = np.sort(np.concatenate([bg_arr, su_arr]))
    arrivals = arrivals[:n_packets + warmup]

    # FIFO deterministic server: departure_i = max(arrival_i, departure_{i-1}) + S
    sojourn = np.empty(len(arrivals))
    prev_dep = 0.0
    for i, a in enumerate(arrivals):
        start = a if a > prev_dep else prev_dep
        dep = start + S_s
        sojourn[i] = (dep - a)
        prev_dep = dep
    sojourn_ms = sojourn[warmup:] * 1000.0 + PROP_MS   # add propagation

    return {
        "mean_ms": float(np.mean(sojourn_ms)),
        "p95_ms":  float(np.percentile(sojourn_ms, 95)),
        "p99_ms":  float(np.percentile(sojourn_ms, 99)),
        "max_ms":  float(np.max(sojourn_ms)),
        "n":       int(len(sojourn_ms)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n_packets", type=int, default=200_000)
    ap.add_argument("--out", type=str, default=os.path.expanduser(
        "results/logs/queue_validation.json"))
    args = ap.parse_args()

    # Operating points from the actual scenarios (combined suit rate, bg).
    # Suit rates: S0~3540, S1~216, S2~185 (representative).
    points = []
    for label, suit_rate in [("S0", 3540.0), ("S1", 216.0), ("S2", 185.0)]:
        for rho_bg in [0.05, 0.10, 0.30, 0.50, 0.70]:
            points.append((label, suit_rate, rho_bg))

    results = []
    print(f"S_ms={S_MS:.4f}  capacity={CAP_MSG_S:.0f} msg/s  prop={PROP_MS} ms\n")
    print(f"{'scn':3} {'suit':>5} {'bg':>5} | {'ana_mean':>8} {'sim_mean':>8} "
          f"{'err%':>6} | {'sim_p95':>8} {'sim_p99':>8}")
    print("-" * 70)
    for label, suit_rate, rho_bg in points:
        # skip overloaded operating points (rho>=0.95): queue diverges, comparison meaningless
        _rho = (suit_rate + rho_bg*CAP_MSG_S) * (S_MS/1000.0)
        if _rho >= 0.95:
            print(f"{label:3} {suit_rate:5.0f} {rho_bg:5.2f} |  (skipped: rho={_rho:.3f} >= 0.95, overloaded)")
            continue
        ana = analytical_mean_ms(suit_rate, rho_bg)
        sims = [simulate(suit_rate, rho_bg, "poisson",
                         n_packets=args.n_packets, seed=s)
                for s in range(args.seeds)]
        sim_mean = float(np.mean([s["mean_ms"] for s in sims]))
        sim_p95  = float(np.mean([s["p95_ms"]  for s in sims]))
        sim_p99  = float(np.mean([s["p99_ms"]  for s in sims]))
        sim_mean_std = float(np.std([s["mean_ms"] for s in sims]))
        err = 100.0 * (sim_mean - ana) / ana
        results.append({
            "scenario": label, "suit_rate": suit_rate, "rho_bg": rho_bg,
            "analytical_mean_ms": round(ana, 4),
            "sim_mean_ms": round(sim_mean, 4),
            "sim_mean_std_ms": round(sim_mean_std, 4),
            "rel_err_pct": round(err, 2),
            "sim_p95_ms": round(sim_p95, 4),
            "sim_p99_ms": round(sim_p99, 4),
        })
        print(f"{label:3} {suit_rate:5.0f} {rho_bg:5.2f} | {ana:8.4f} {sim_mean:8.4f} "
              f"{err:6.2f} | {sim_p95:8.4f} {sim_p99:8.4f}")

    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {args.out}")

    errs = [abs(r["rel_err_pct"]) for r in results]
    print(f"\nMean |rel error| analytical-vs-sim: {np.mean(errs):.2f}%  "
          f"(max {np.max(errs):.2f}%)")
    print("Validation PASSES if mean error < ~5%.")


if __name__ == "__main__":
    main()
