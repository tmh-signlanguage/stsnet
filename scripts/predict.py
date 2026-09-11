"""
Run STS-Net v0.2 (ClipClassifier) inference on a .pose or video file.

Two prediction modes:
  - Per-frame (default): the classification heads are applied to every frame's
    fusion features, producing one label per head per frame.
  - Clip-level (--sign_start/--sign_end): attention is pooled over the given
    sign window and each head predicts a single label for the whole clip.

Either mode can additionally dump raw per-frame encoder features (T, 256) to
a .pt file with --embeddings, for use as input to a downstream model.

Usage — per-frame predictions, CSV to stdout:
    python scripts/predict.py clip.pose --ckpt checkpoints/stsnet_v02.pt
    python scripts/predict.py clip.mp4  --ckpt checkpoints/stsnet_v02.pt

Usage — clip-level prediction for a single sign:
    python scripts/predict.py clip.pose \\
        --ckpt checkpoints/stsnet_v02.pt \\
        --sign_start 12 --sign_end 58

Usage — write CSV to a file and dump per-frame embeddings:
    python scripts/predict.py clip.pose \\
        --out preds.csv --embeddings feats.pt

Output format (per-frame, default):
    frame,shape,att,cloc,ctype,motion,hand_type,nondom_shape,nondom_att
    0,Flata handen,vänsterriktad-nedåtvänd,none,none,none,one,...

Output format (clip-level, --sign_start/--sign_end):
    shape,att,cloc,ctype,motion,hand_type,nondom_shape,nondom_att
    Flata handen,vänsterriktad-nedåtvänd,none,none,none,one,...
"""

import argparse
import csv
import subprocess
import sys
import tempfile
import time
from pathlib import Path


VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}


def extract_pose(video_path: Path, pose_path: Path) -> None:
    """Run video_to_pose (MediaPipe Holistic) on a video file."""
    print(f"Extracting pose from {video_path.name}...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ["video_to_pose", "-i", str(video_path), "-o", str(pose_path),
         "--format", "mediapipe"],
        capture_output=True,
    )
    if result.returncode != 0:
        print()
        msg = result.stderr.decode()[-400:]
        print(f"Error: pose extraction failed:\n{msg}", file=sys.stderr)
        sys.exit(1)
    print(f"done ({time.time() - t0:.1f}s)")


def write_csv(rows: list[list], headers: list[str], out_path: str | None) -> None:
    f = open(out_path, "w", newline="") if out_path else sys.stdout
    try:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    finally:
        if out_path:
            f.close()


def main():
    ap = argparse.ArgumentParser(
        description="STS-Net v0.2 inference on a .pose or video file")
    ap.add_argument("input_file", help="Input .pose file or video (.mp4, .mov, ...)")
    ap.add_argument("--ckpt", default="checkpoints/stsnet_v02.pt",
                    help="Checkpoint path (default: checkpoints/stsnet_v02.pt)")
    ap.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; use cuda for GPU)")
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--no_z", action="store_true",
                    help="Strip z-coordinate from pose streams (2D input). "
                         "Auto-detected from the checkpoint when not given.")
    # Clip-level mode
    ap.add_argument("--sign_start", type=int, default=None,
                    help="First frame of signing window (clip-level mode)")
    ap.add_argument("--sign_end",   type=int, default=None,
                    help="Last frame (exclusive) of signing window (clip-level mode)")
    # Per-frame output options
    ap.add_argument("--heads", nargs="+", default=None,
                    help="Heads to include (default: all heads present in the checkpoint)")
    ap.add_argument("--start", type=int, default=None,
                    help="First frame to include in per-frame output")
    ap.add_argument("--end",   type=int, default=None,
                    help="Last frame (exclusive) to include in per-frame output")
    # Output
    ap.add_argument("--out", default=None,
                    help="CSV output path (default: stdout)")
    ap.add_argument("--embeddings", default=None,
                    help="Save per-frame encoder features (T, 256) to this .pt path")
    args = ap.parse_args()

    input_path = Path(args.input_file)
    if not input_path.exists():
        print(f"Error: input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"Error: checkpoint not found: {ckpt_path}", file=sys.stderr)
        sys.exit(1)

    if (args.sign_start is None) != (args.sign_end is None):
        print("Error: --sign_start and --sign_end must be given together",
              file=sys.stderr)
        sys.exit(1)

    # ── Pose extraction (video input) ───────────────────────────────────────
    tmp_dir = None
    if input_path.suffix.lower() in VIDEO_SUFFIXES:
        tmp_dir = tempfile.TemporaryDirectory()
        pose_path = Path(tmp_dir.name) / (input_path.name + ".pose")
        extract_pose(input_path, pose_path)
    else:
        pose_path = input_path

    try:
        # ── Model loading ────────────────────────────────────────────────────
        print("Loading model...", end=" ", file=sys.stderr, flush=True)
        t0 = time.time()
        from stsnet.inference import ClipClassifierInference
        model = ClipClassifierInference(
            ckpt_path, device=args.device,
            no_z=True if args.no_z else None,
        )
        print(f"done ({time.time() - t0:.1f}s)", file=sys.stderr)

        # ── Predictions ──────────────────────────────────────────────────────
        if args.sign_start is not None:
            # Clip-level mode: one label per head for the whole sign window.
            pred = model.predict_phonology(
                pose_path, args.sign_start, args.sign_end,
                handedness=args.handedness,
            )
            if pred is None:
                print("Error: failed to load pose file", file=sys.stderr)
                sys.exit(1)
            heads = [h for h in args.heads if h in pred] if args.heads else list(pred.keys())
            write_csv([[pred[h] for h in heads]], heads, args.out)
        else:
            # Per-frame mode: one label per head per frame.
            preds = model.predict_frames(pose_path, handedness=args.handedness)
            if preds is None:
                print("Error: failed to load pose file", file=sys.stderr)
                sys.exit(1)
            heads = [h for h in args.heads if h in preds] if args.heads else list(preds.keys())
            T = len(next(iter(preds.values())))
            t0f = args.start or 0
            t1f = args.end if args.end is not None else T
            rows = [[t] + [preds[h][t] for h in heads] for t in range(t0f, t1f)]
            write_csv(rows, ["frame"] + heads, args.out)

        # ── Embeddings ───────────────────────────────────────────────────────
        if args.embeddings is not None:
            import torch
            feats = model.frame_features(pose_path, handedness=args.handedness)
            if feats is None:
                print("Error: failed to load pose file", file=sys.stderr)
                sys.exit(1)
            torch.save(torch.from_numpy(feats), args.embeddings)
            print(f"Saved per-frame embeddings {tuple(feats.shape)} to {args.embeddings}",
                  file=sys.stderr)
    finally:
        if tmp_dir:
            tmp_dir.cleanup()


if __name__ == "__main__":
    main()
