"""
Gymnasium environment for the RL sampling agent (S2) — PATH B: multi-suit cell.

Setting: M telepresence suits share ONE 6G URLLC tactile slice of capacity C.
By the mean-field assumption all suits run the SAME policy, so the per-joint
send probability p determines the AGGREGATE slice load:
    R_aggregate = M * p * R_PER_SUIT      (msg/s)
Thus a single agent's policy strongly controls the shared-channel latency.

Tension (the whole point of Path B): under congestion (large M) a fixed
JND-deadband keeps p constant and OVERLOADS the slice; the RL agent learns to
push p BELOW the JND-optimal rate — sacrificing some per-suit fidelity — to keep
the shared slice within the 1 ms tactile budget. A fixed deadband cannot do this.

State (8-dim):
  [0] abs delta since last send (deg)
  [1] velocity (deg/frame)
  [2] acceleration (deg/frame^2)
  [3] time since last send (s, capped)
  [4] current reconstruction error estimate (deg)
  [5] congestion = M / M_MAX  (network-signalled number of competing suits)
  [6] remaining latency budget (normalised, from realised aggregate load)
  [7] phase indicator (0=slow, 1=fast)

Action: 0 = skip, 1 = send
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces

# Per-suit always-on combined rate: body 24@60 + hand 21@100 = 3540 msg/s
R_PER_SUIT = 3540.0
# Max number of concurrently active suits considered (for normalisation & sampling)
M_MAX = 50


class SuitSamplingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, angle_traces: np.ndarray, channel_model,
                 fps: float = 60.0,
                 latency_budget_ms: float = 1.0,
                 jnd_deg: float = 2.0,
                 lambda_traffic: float = 1.0,
                 lambda_distortion: float = 5.0,
                 lambda_latency: float = 5.0,
                 m_range=(1, M_MAX),
                 episode_len: int = 1500,
                 blind_channel: bool = False):
        super().__init__()
        self.traces         = angle_traces
        self.channel        = channel_model
        self.fps            = fps
        self.latency_budget = latency_budget_ms
        self.jnd            = jnd_deg
        self.lam_t          = lambda_traffic
        self.lam_d          = lambda_distortion
        self.lam_l          = lambda_latency
        self.m_lo, self.m_hi = m_range
        self.ep_len         = episode_len
        self.blind          = blind_channel   # ablation: hide channel state from obs

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)

        self._T = self.traces.shape[0]
        self._J = self.traces.shape[1]
        self._rng = np.random.default_rng()

    # ---- mean-field aggregate latency: M suits each sending at fraction p ----
    def _latency_ms(self, p_send: float) -> float:
        self.channel.cfg.background_load = 0.0          # all load is suit traffic
        aggregate_rate = self._M * p_send * R_PER_SUIT
        return self.channel.mean_latency_ms(aggregate_rate)

    def _obs(self) -> np.ndarray:
        cur   = self.traces[self._t, self._joint]
        prev  = self.traces[max(0, self._t - 1), self._joint]
        prev2 = self.traces[max(0, self._t - 2), self._joint]
        delta = abs(cur - self._last_sent_angle)
        vel   = abs(cur - prev)
        acc   = abs((cur - prev) - (prev - prev2))
        steps = max(self._t - self._t0, 1)
        p_send = self._msg_count / steps
        time_since = (self._t - self._last_sent_t) / self.fps
        lat = self._latency_ms(p_send)
        lat_budget_rem = np.clip(
            (self.latency_budget - lat) / self.latency_budget, -1.0, 1.0)
        congestion = self._M / float(M_MAX)
        if self.blind:
            congestion = 0.0; lat_budget_rem = 0.0   # ablation: blind to channel
        phase = 1.0 if vel > self.jnd else 0.0
        return np.array([delta, vel, acc, min(time_since, 10.0),
                         delta, congestion, lat_budget_rem, phase],
                        dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._t0 = int(self._rng.integers(0, max(1, self._T - self.ep_len - 2)))
        self._t  = self._t0
        self._joint = int(self._rng.integers(0, self._J))
        self._M = int(self._rng.integers(self.m_lo, self.m_hi + 1))   # # competing suits
        self._last_sent_angle = self.traces[self._t, self._joint]
        self._last_sent_t     = self._t
        self._msg_count       = 0
        return self._obs(), {}

    def step(self, action):
        cur = self.traces[self._t, self._joint]
        reward = 0.0
        steps = max(self._t - self._t0, 1)

        if action == 1:                       # send
            self._msg_count += 1
            reward -= self.lam_t * 1.0
            p_send = self._msg_count / steps
            lat = self._latency_ms(p_send)
            if lat > self.latency_budget:
                # clip per-step latency penalty so overload does not explode training
                reward -= self.lam_l * min(lat - self.latency_budget, 5.0)
            self._last_sent_angle = cur
            self._last_sent_t     = self._t
        else:                                 # skip
            err = abs(cur - self._last_sent_angle)
            if err > self.jnd:
                reward -= self.lam_d * (err - self.jnd)

        self._t += 1
        done = (self._t >= self._T - 1) or (self._t - self._t0 >= self.ep_len)
        return self._obs(), reward, done, False, {}

    def render(self): pass
