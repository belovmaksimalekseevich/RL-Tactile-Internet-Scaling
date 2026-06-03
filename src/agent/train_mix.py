"""Train a JOINT body+hand policy: each episode samples a random stream
(body @60Hz or hand @100Hz). Compares against the body-only policy (ppo_s2_B)."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))
import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.loaders.dip_loader import load_subjects as load_dip
from src.loaders.ninapro_loader import load_subjects as load_ninapro
from src.network.channel import ChannelModel, ChannelConfig
from src.agent.env import SuitSamplingEnv

CHECKPOINTS_DIR = Path(__file__).parents[2] / "results/checkpoints"
SLICE_BW_BPS = 12.8e6


class MixEnv(SuitSamplingEnv):
    """Per-episode random choice of stream (traces, fps)."""
    def __init__(self, streams, channel_model, **kw):
        super().__init__(angle_traces=streams[0][0], channel_model=channel_model,
                         fps=streams[0][1], **kw)
        self._streams = streams

    def reset(self, seed=None, options=None):
        idx = int(self._rng.integers(0, len(self._streams)))
        self.traces, self.fps = self._streams[idx]
        self._T = self.traces.shape[0]
        self._J = self.traces.shape[1]
        return super().reset(seed=seed, options=options)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600_000)
    ap.add_argument("--out", type=str, default="ppo_s2_mix")
    args = ap.parse_args()

    print("Loading body (DIP s_01,s_02) + hand (NinaPro s_1,s_2)...")
    body = np.vstack([s["angles_deg"] for s in load_dip(["s_01", "s_02"])])
    hand = np.vstack([s["angles_deg"] for s in load_ninapro(["s_1", "s_2"])])
    print(f"  body {body.shape} @60Hz, hand {hand.shape} @100Hz")
    streams = [(body, 60.0), (hand, 100.0)]

    def make_env():
        ch = ChannelModel(ChannelConfig(bandwidth_bps=SLICE_BW_BPS, background_load=0.0))
        return MixEnv(streams, channel_model=ch, jnd_deg=2.0,
                      lambda_traffic=1.0, lambda_distortion=5.0, lambda_latency=5.0,
                      m_range=(1, 50), episode_len=1500)

    env = make_vec_env(make_env, n_envs=4)
    model = PPO("MlpPolicy", env, learning_rate=3e-4, n_steps=2048, batch_size=256,
                n_epochs=10, gamma=0.99, verbose=1, tensorboard_log=None)
    print(f"Training PPO (mix) for {args.steps} steps...")
    model.learn(total_timesteps=args.steps)
    model.save(str(CHECKPOINTS_DIR / f"{args.out}.zip"))
    print(f"Model saved to {CHECKPOINTS_DIR / (args.out + '.zip')}")


if __name__ == "__main__":
    main()
