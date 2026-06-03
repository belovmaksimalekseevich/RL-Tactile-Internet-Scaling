"""
Train S2 RL agent with PPO — PATH B: multi-suit shared 6G tactile slice.

Usage:
  cd ~/suit-sim && source venv/bin/activate
  python src/agent/train.py --subjects s_01 s_02 --steps 600000 --m_hi 50
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

import argparse
from pathlib import Path
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import CheckpointCallback

from src.loaders.dip_loader import load_subjects
from src.network.channel import ChannelModel, ChannelConfig
from src.agent.env import SuitSamplingEnv

CHECKPOINTS_DIR = Path(__file__).parents[2] / "results/checkpoints"
CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

# Shared tactile slice capacity (must match experiments): 12.8 Mbps -> 8000 msg/s
SLICE_BW_BPS = 12.8e6


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", nargs="+", default=["s_01", "s_02"])
    ap.add_argument("--steps",   type=int,   default=600_000)
    ap.add_argument("--jnd",     type=float, default=2.0)
    ap.add_argument("--m_lo",    type=int,   default=1)
    ap.add_argument("--m_hi",    type=int,   default=50)
    ap.add_argument("--ep_len",  type=int,   default=1500)
    ap.add_argument("--lam_t",   type=float, default=1.0)
    ap.add_argument("--lam_d",   type=float, default=5.0)
    ap.add_argument("--lam_l",   type=float, default=5.0)
    ap.add_argument("--out",     type=str,   default="ppo_s2_B")
    ap.add_argument("--blind",   action="store_true", help="ablation: hide channel state")
    args = ap.parse_args()

    print("Loading training data...")
    sequences = load_subjects(args.subjects)
    all_angles = np.vstack([s["angles_deg"] for s in sequences])
    print(f"  Total frames: {all_angles.shape[0]}, joints: {all_angles.shape[1]}")
    print(f"  PATH B: multi-suit slice {SLICE_BW_BPS/1e6:.1f} Mbps, "
          f"M ~ U[{args.m_lo}, {args.m_hi}] competing suits, ep_len={args.ep_len}")

    def make_env():
        channel = ChannelModel(ChannelConfig(bandwidth_bps=SLICE_BW_BPS,
                                             background_load=0.0))
        return SuitSamplingEnv(
            angle_traces      = all_angles,
            channel_model     = channel,
            jnd_deg           = args.jnd,
            lambda_traffic    = args.lam_t,
            lambda_distortion = args.lam_d,
            lambda_latency    = args.lam_l,
            m_range           = (args.m_lo, args.m_hi),
            episode_len       = args.ep_len,
            blind_channel     = args.blind,
        )

    env = make_vec_env(make_env, n_envs=4)

    model = PPO(
        "MlpPolicy", env,
        learning_rate=3e-4, n_steps=2048, batch_size=256,
        n_epochs=10, gamma=0.99, verbose=1, tensorboard_log=None,
    )

    checkpoint_cb = CheckpointCallback(
        save_freq=100_000, save_path=str(CHECKPOINTS_DIR), name_prefix=args.out)

    print(f"Training PPO for {args.steps} steps...")
    model.learn(total_timesteps=args.steps, callback=checkpoint_cb)

    final_path = CHECKPOINTS_DIR / f"{args.out}.zip"
    model.save(str(final_path))
    print(f"Model saved to {final_path}")


if __name__ == "__main__":
    main()
