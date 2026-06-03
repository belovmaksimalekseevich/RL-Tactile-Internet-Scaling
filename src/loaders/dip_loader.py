"""
DIP-IMU dataset loader.

Loads pkl files from DIP_IMU and returns joint angles (degrees)
derived from SMPL axis-angle ground-truth poses.

Output format: list of dicts per sequence:
  {
    "subject":   str,          # e.g. "s_01"
    "seq":       str,          # e.g. "01"
    "fps":       int,          # 60
    "n_frames":  int,
    "angles_deg": np.ndarray,  # (n_frames, 72) SMPL axis-angle -> degrees
    "imu_ori":   np.ndarray,   # (n_frames, 17, 3, 3) raw orientations
    "imu_acc":   np.ndarray,   # (n_frames, 17, 3) raw accelerations
  }
"""
import os
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict

DIP_IMU_DIR = Path(__file__).parents[2] / "data/raw/dip/dip_imu/DIP_IMU"
FPS = 60

# 17 IMU locations for reference
IMU_NAMES = [
    "head","spine2","belly","lchest","rchest",
    "lshoulder","rshoulder","lelbow","relbow",
    "lhip","rhip","lknee","rknee",
    "lwrist","rwrist","lankle","rankle"
]

def _axis_angle_to_degrees(aa: np.ndarray) -> np.ndarray:
    """Convert axis-angle (N,72) -> magnitude in degrees per joint (N,24)."""
    aa = aa.reshape(-1, 24, 3)
    magnitudes = np.linalg.norm(aa, axis=-1)          # (N, 24) radians
    return np.degrees(magnitudes)                      # (N, 24) degrees

def load_subject(subject: str, data_dir: Path = DIP_IMU_DIR) -> List[Dict]:
    """Load all sequences for one subject. subject e.g. s_01."""
    subj_dir = data_dir / subject
    if not subj_dir.exists():
        raise FileNotFoundError(f"Subject dir not found: {subj_dir}")
    sequences = []
    for pkl_file in sorted(subj_dir.glob("*.pkl")):
        with open(pkl_file, "rb") as f:
            data = pickle.load(f, encoding="latin1")
        gt = data["gt"]          # (T, 72) axis-angle ground truth
        sequences.append({
            "subject":    subject,
            "seq":        pkl_file.stem,
            "fps":        FPS,
            "n_frames":   gt.shape[0],
            "angles_deg": _axis_angle_to_degrees(gt),  # (T, 24) deg
            "angles_raw": gt,                           # (T, 72) raw axis-angle
            "imu_ori":    data["imu_ori"],              # (T, 17, 3, 3)
            "imu_acc":    data["imu_acc"],              # (T, 17, 3)
        })
    return sequences

def load_subjects(subjects: List[str] = None, data_dir: Path = DIP_IMU_DIR) -> List[Dict]:
    """Load multiple subjects. Default: s_01, s_02, s_03."""
    if subjects is None:
        subjects = ["s_01", "s_02", "s_03"]
    all_seqs = []
    for s in subjects:
        all_seqs.extend(load_subject(s, data_dir))
        print(f"  Loaded {s}: {len(all_seqs)} sequences total")
    return all_seqs

if __name__ == "__main__":
    seqs = load_subjects(["s_01"])
    s = seqs[0]
    print(f"Subject: {s['subject']}, seq: {s['seq']}, frames: {s['n_frames']}, fps: {s['fps']}")
    print(f"angles_deg shape: {s['angles_deg'].shape}")  # (T, 24)
    print(f"angles range: {s['angles_deg'].min():.1f} .. {s['angles_deg'].max():.1f} deg")
