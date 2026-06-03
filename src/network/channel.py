"""
Network channel model: M/G/1 queue + jitter + packet loss.

Models a shared 6G uplink with multiple competing flows.
Used to compute realistic end-to-end latency for each scenario.

Key formula (M/G/1 mean waiting time, Pollaczek-Khinchine):
  W = lambda * E[S^2] / (2*(1-rho))   where rho = lambda * E[S]
"""
import numpy as np
from dataclasses import dataclass

@dataclass
class ChannelConfig:
    # Uplink capacity
    bandwidth_bps: float = 10e6         # 10 Mbps: dedicated 6G URLLC tactile slice for one suit (always-on ~57% load)
    # Packet parameters
    payload_bytes: int = 200            # ~24 floats + header
    # Competing background load (fraction of capacity)
    background_load: float = 0.3        # 30% background traffic
    # Additional one-way propagation delay
    prop_delay_ms: float = 0.1          # 0.1 ms (local 6G)
    # Random jitter std dev
    jitter_std_ms: float = 0.05
    # Packet loss probability
    loss_prob: float = 0.001

class ChannelModel:
    """
    Computes end-to-end latency given message rate (msg/s).
    
    Returns latency in milliseconds.
    """
    def __init__(self, cfg: ChannelConfig = None):
        self.cfg = cfg or ChannelConfig()

    def _service_time_ms(self) -> float:
        """Transmission time for one packet in ms."""
        bits = self.cfg.payload_bytes * 8
        return (bits / self.cfg.bandwidth_bps) * 1000.0

    def mean_latency_ms(self, msg_rate_per_sec: float) -> float:
        """
        Mean E2E latency for given message rate using M/G/1 model.
        msg_rate_per_sec: total messages/s from suit sensors.
        """
        S = self._service_time_ms()          # mean service time ms
        # Total arrival rate including background
        capacity_msg_per_s = (self.cfg.bandwidth_bps /
                              (self.cfg.payload_bytes * 8))
        bg_rate = self.cfg.background_load * capacity_msg_per_s
        lam = msg_rate_per_sec + bg_rate     # total lambda

        rho = lam * (S / 1000.0)            # utilisation (dimensionless)
        rho = min(rho, 0.999)               # cap to avoid singularity at rho->1

        # Exact M/D/1 mean waiting time in queue (deterministic service).
        # Pollaczek-Khinchine with E[S^2]=S^2 reduces to: W = rho*S / (2(1-rho))
        # NOTE: S is in ms here, so W is in ms (units consistent). The previous
        # version mixed lam [1/s] with S^2 [ms^2], inflating W by up to ~1000x.
        W_queue = (rho * S) / (2.0 * (1.0 - rho))
        # Total latency = queue wait + service + propagation
        latency = W_queue + S + self.cfg.prop_delay_ms
        return latency

    def sample_latency_ms(self, msg_rate_per_sec: float,
                          n_samples: int = 1, rng: np.random.Generator = None) -> np.ndarray:
        """Sample latency values with jitter."""
        rng = rng or np.random.default_rng()
        mean = self.mean_latency_ms(msg_rate_per_sec)
        jitter = rng.normal(0, self.cfg.jitter_std_ms, n_samples)
        return np.maximum(mean + jitter, 0.0)

    def is_packet_lost(self, rng: np.random.Generator = None) -> bool:
        rng = rng or np.random.default_rng()
        return rng.random() < self.cfg.loss_prob

if __name__ == "__main__":
    ch = ChannelModel()
    for rate in [100, 500, 1000, 2000, 5000]:
        print(f"rate={rate:5d} msg/s  latency={ch.mean_latency_ms(rate):.3f} ms")
