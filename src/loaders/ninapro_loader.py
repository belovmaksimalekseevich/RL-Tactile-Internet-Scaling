"""
NinaPro DB9 loader.

Loads S*_E2_A1.mat and S*_E3_A1.mat per subject.
Returns calibrated joint angles (degrees) for 21 valid channels.

Sensor 10 (0-indexed) = MCP2-3_A is excluded (all-NaN, known hardware issue).

Output format: list of dicts per file:
  {
    "subject":    str,           # e.g. "s_1"
    "exercise":   str,           # "E2" or "E3"
    "fps":        int,           # 100
    "n_frames":   int,
    "angles_deg": np.ndarray,    # (T, 21) calibrated angles, clipped [-180, 180]
    "stimulus":   np.ndarray,    # (T,) movement labels (0=rest)
    "n_movements": int,
  }
"""
import os
import numpy as np
import scipy.io as sio
from pathlib import Path
from typing import List, Dict

NINAPRO_DIR = Path(__file__).parents[2] / "data/raw/ninapro_db9"
FPS = 100
VALID_COLS = list(range(10)) + list(range(11, 22))   # drop col 10 (sensor 11, NaN)
CLIP_DEG   = 180.0   # physiological limit: clip outliers

# Joint names for the 21 valid channels (DB9 / CyberGlove II)
JOINT_NAMES = [
    "Thumb_CMC_ab", "Thumb_CMC_flex", "Thumb_MCP", "Thumb_DIP",
    "Index_MCP", "Index_PIP",
    "Middle_MCP", "Middle_PIP",
    "MCP2_3",          # sensor 9 (index 9)
    # sensor 10 SKIPPED
    "Ring_MCP", "Ring_PIP",
    "Pinky_MCP", "Pinky_PIP",
    "MCP3_4", "MCP4_5",
    "Wrist_flex", "Wrist_dev", "Wrist_rot",
    "Palm_arch",
    "Thumb_IP", "Index_DIP",
]

def _load_mat(path: str) -> Dict:
    mat = sio.loadmat(path)
    angles_full = mat["angles"]                           # (T, 22)
    angles = angles_full[:, VALID_COLS].astype(np.float32)  # (T, 21)
    # Replace remaining NaNs with 0 (rare, edge frames)
    nan_mask = np.isnan(angles)
    if nan_mask.any():
        angles[nan_mask] = 0.0
    # Clip physiological outliers (calibration artefacts)
    angles = np.clip(angles, -CLIP_DEG, CLIP_DEG)
    stimulus = mat["restimulus"].flatten().astype(np.int32)
    return angles, stimulus

def load_subject(subject: str, data_dir: Path = NINAPRO_DIR) -> List[Dict]:
    """
    Load E2 and E3 files for one subject.
    subject: e.g. s_1 -> looks for s_1_angles/s_1_angles/S1_E2_A1.mat
    """
    idx = subject.split("_")[1]           # 1, 2, 3
    subj_dir = data_dir / f"s_{idx}_angles" / f"s_{idx}_angles"
    if not subj_dir.exists():
        raise FileNotFoundError(f"NinaPro subject dir not found: {subj_dir}")

    results = []
    for ex in ["E2", "E3"]:
        fname = subj_dir / f"S{idx}_{ex}_A1.mat"
        if not fname.exists():
            print(f"  [warn] {fname} not found, skipping")
            continue
        angles, stimulus = _load_mat(str(fname))
        results.append({
            "subject":    subject,
            "exercise":   ex,
            "fps":        FPS,
            "n_frames":   angles.shape[0],
            "angles_deg": angles,            # (T, 21) float32 degrees
            "stimulus":   stimulus,          # (T,) int  movement label
            "n_movements": int(stimulus.max()),
        })
        print(f"  {subject}/{ex}: {angles.shape[0]} frames, "
              f"{int(stimulus.max())} movements, "
              f"range [{angles.min():.1f}, {angles.max():.1f}] deg")
    return results

def load_subjects(subjects: List[str] = None,
                  data_dir: Path = NINAPRO_DIR) -> List[Dict]:
    """Load multiple subjects. Default: s_1, s_2, s_3."""
    if subjects is None:
        subjects = ["s_1", "s_2", "s_3"]
    all_seqs = []
    for s in subjects:
        seqs = load_subject(s, data_dir)
        all_seqs.extend(seqs)
    print(f"Total NinaPro sequences: {len(all_seqs)}")
    return all_seqs

if __name__ == "__main__":
    seqs = load_subjects(["s_1"])
    for s in seqs:
        print(f"\n{s[subject]}/{s[exercise]}: "
              f"{s[n_frames]} frames @ {s[fps]} Hz, "
              f"{s[angles_deg].shape[1]} joints")
