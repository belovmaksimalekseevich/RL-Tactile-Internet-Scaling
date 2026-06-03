"""
Metrics collector for one experiment run.

Collects per-message data and computes aggregate statistics:
  - Traffic: total messages, bytes/s, reduction vs S0
  - Latency: mean/p95/p99 E2E latency (ms)
  - Energy: estimated ESP32 radio energy (uJ)
  - Distortion: RMSE between received and ground-truth angles (degrees)
"""
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional

# ESP32 radio energy model (from datasheet + literature)
# Source: ESP32 Technical Reference Manual
E_TX_uJ_per_packet = 0.5     # ~0.5 uJ per packet transmission
E_IDLE_uJ_per_ms   = 0.02    # idle listening energy

@dataclass
class MessageRecord:
    t_send:     float          # send timestamp (s)
    t_recv:     float          # receive timestamp (s)
    seq:        int
    sensor_id:  int
    angles_sent: np.ndarray    # (24,) or (21,) degrees
    angles_gt:   np.ndarray    # ground truth at same frame
    lost:        bool = False

class MetricsCollector:
    def __init__(self, scenario: str, fps: float, n_sensors: int,
                 payload_bytes: int = 200):
        self.scenario      = scenario
        self.fps           = fps
        self.n_sensors     = n_sensors
        self.payload_bytes = payload_bytes
        self.records: List[MessageRecord] = []
        self._total_frames = 0

    def record(self, rec: MessageRecord):
        self.records.append(rec)

    def set_total_frames(self, n: int):
        self._total_frames = n

    # ------------------------------------------------------------------ #
    def traffic_stats(self):
        n_sent = sum(1 for r in self.records if not r.lost)
        n_total_possible = self._total_frames * self.n_sensors
        duration_s = (self.records[-1].t_send - self.records[0].t_send
                      if len(self.records) > 1 else 1.0)
        duration_s = max(duration_s, 1e-9)
        return {
            "n_sent":           n_sent,
            "n_possible":       n_total_possible,
            "reduction_pct":    100.0 * (1 - n_sent / max(n_total_possible, 1)),
            "msg_per_sec":      n_sent / duration_s,
            "bytes_per_sec":    n_sent * self.payload_bytes / duration_s,
        }

    def latency_stats(self):
        latencies = [(r.t_recv - r.t_send) * 1000
                     for r in self.records if not r.lost]
        if not latencies:
            return {}
        lat = np.array(latencies)
        return {
            "mean_ms":  float(np.mean(lat)),
            "std_ms":   float(np.std(lat)),
            "p50_ms":   float(np.percentile(lat, 50)),
            "p95_ms":   float(np.percentile(lat, 95)),
            "p99_ms":   float(np.percentile(lat, 99)),
            "max_ms":   float(np.max(lat)),
        }

    def energy_stats(self):
        n_sent = sum(1 for r in self.records if not r.lost)
        duration_s = (self.records[-1].t_send - self.records[0].t_send
                      if len(self.records) > 1 else 1.0)
        tx_energy_uJ  = n_sent * E_TX_uJ_per_packet
        idle_energy_uJ = duration_s * 1000 * E_IDLE_uJ_per_ms  # ms * uJ/ms
        return {
            "tx_energy_uJ":    tx_energy_uJ,
            "idle_energy_uJ":  idle_energy_uJ,
            "total_energy_uJ": tx_energy_uJ + idle_energy_uJ,
        }

    def distortion_stats(self, jnd_deg: float = 2.0):
        """
        RMSE between sent angles and ground-truth.
        JND threshold: ~2 deg for body motion perception
        (Ref: Pongrac 2008, sensory threshold for motion perception).
        """
        if not self.records:
            return {}
        errors = []
        for r in self.records:
            if not r.lost and r.angles_gt is not None:
                errors.append(np.sqrt(np.mean((r.angles_sent - r.angles_gt)**2)))
        if not errors:
            return {}
        e = np.array(errors)
        return {
            "rmse_mean_deg":      float(np.mean(e)),
            "rmse_max_deg":       float(np.max(e)),
            "pct_above_jnd":      float(100.0 * np.mean(e > jnd_deg)),
            "jnd_threshold_deg":  jnd_deg,
        }

    def summary(self) -> dict:
        return {
            "scenario":  self.scenario,
            "traffic":   self.traffic_stats(),
            "latency":   self.latency_stats(),
            "energy":    self.energy_stats(),
            "distortion":self.distortion_stats(),
        }
