"""
Interactive activation inspector for STS-Net v0.2 (ClipClassifier).

Takes a list of video (or .pose) files on the command line, runs each
through the model, and serves a web GUI showing the video alongside
per-frame activation heatmaps for every head, plus the attention trace —
synced to the video playhead. No CSV / dataset metadata required.

Additional clips can be added at any time by dragging video files onto the
browser window — they're uploaded, pose-extracted in the background (with a
progress indicator), and appear in the sidebar once ready.

Usage:
    python scripts/inspector.py clip1.mp4 clip2.mp4 clip3.pose \\
        --ckpt checkpoints/stsnet_v02.pt
"""

import argparse
import mimetypes
import re
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from flask import Flask, Response, jsonify, render_template_string, request

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024 * 1024  # 2 GB upload cap

VIDEO_SUFFIXES = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
UPLOAD_SUFFIXES = VIDEO_SUFFIXES | {".pose"}

# ---------------------------------------------------------------------------
# Global state (populated in main())
# ---------------------------------------------------------------------------
CLIPS: list[dict] = []          # [{idx, name, video_path, pose_path, status}]
CLIPS_LOCK = threading.Lock()   # guards CLIPS during background uploads
MODEL   = None
VOCAB: dict = {}                # idx_to_shape/att/motion/cloc/ctype
DEVICE  = torch.device("cpu")
HANDEDNESS = "right"
ACT_CACHE: dict[int, dict] = {}
KP_CACHE: dict[int, dict] = {}      # raw keypoint overlay data, see _load_raw_keypoints
UPLOAD_DIR: Path | None = None      # where drag-and-dropped files are saved
POSE_CACHE_DIR: Path | None = None  # where their extracted .pose files go


# ---------------------------------------------------------------------------
# Pose extraction
# ---------------------------------------------------------------------------

def extract_pose(video_path: Path, pose_path: Path) -> bool:
    """Run video_to_pose (MediaPipe Holistic) on a video file. Returns success."""
    print(f"  Extracting pose from {video_path.name}...", end=" ", flush=True)
    t0 = time.time()
    result = subprocess.run(
        ["video_to_pose", "-i", str(video_path), "-o", str(pose_path),
         "--format", "mediapipe"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"FAILED\n    {result.stderr.decode()[-300:]}")
        return False
    print(f"done ({time.time() - t0:.1f}s)")
    return True


def _probe_edit_list_offset(video_path: Path) -> float:
    """
    Seconds of pre-roll hidden by an MP4 edit list before the video's first
    *displayed* frame (0.0 if there is none).

    This shows up when a clip is trimmed mid-stream with `ffmpeg -c copy`:
    since the cut point usually isn't on a keyframe, ffmpeg copies packets
    starting from the preceding keyframe, then adds an edit list telling
    compliant players to skip the extra leading frames. Browsers honor
    that (video.currentTime=0 is the intended cut point), but video_to_pose
    decodes every raw frame in the file — so its frame 0 is actually this
    many seconds *before* what the browser shows at time 0, and the two
    only line back up at the very end of the clip (ffprobe's reported
    stream `start_time` already reflects this offset).
    """
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=start_time",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, timeout=10,
        )
        return max(0.0, float(result.stdout.decode().strip()))
    except Exception:
        return 0.0


def _load_raw_keypoints(pose_path: Path, video_path: Path | None = None) -> dict | None:
    """
    Raw, un-normalized MediaPipe landmarks for the on-video keypoint overlay:
    full body pose (33 pts), both hands (21 pts each), and a reduced set of
    face points — in original-video pixel coordinates.

    Unlike stsnet.data.pose_io.load_pose_streams (used for model input),
    this skips pose.normalize() and the wrist-centering/shoulder-scaling
    steps, so the coordinates line up 1:1 with the source video frame
    (modulo the edit-list offset corrected for below).
    """
    from pose_format import Pose
    from stsnet.data.pose_io import (
        masked_to_float, POSE_LANDMARK_SLICE, LEFT_HAND_SLICE,
        RIGHT_HAND_SLICE, FACE_INDICES,
    )

    try:
        with open(pose_path, "rb") as f:
            pose = Pose.read(f.read())
    except Exception:
        return None

    data = pose.body.data[:, 0, :, :]      # (T, 576, 3)
    conf = pose.body.confidence[:, 0, :]   # (T, 576)

    body_pts  = masked_to_float(data[:, POSE_LANDMARK_SLICE, :], conf[:, POSE_LANDMARK_SLICE])
    left_pts  = masked_to_float(data[:, LEFT_HAND_SLICE,     :], conf[:, LEFT_HAND_SLICE])
    right_pts = masked_to_float(data[:, RIGHT_HAND_SLICE,    :], conf[:, RIGHT_HAND_SLICE])
    face_pts  = masked_to_float(data[:, FACE_INDICES,        :], conf[:, FACE_INDICES])

    def _xy(arr):
        # (T, N, 3) -> (T, N) list of [x, y] pixel coords (rounded), or None
        # for points MediaPipe didn't track that frame.
        out = []
        for frame in arr[..., :2]:
            out.append([
                None if np.isnan(x) or np.isnan(y) else [round(float(x), 1), round(float(y), 1)]
                for x, y in frame
            ])
        return out

    T   = int(body_pts.shape[0])
    fps = float(pose.body.fps)

    frame_offset = 0
    if video_path is not None:
        offset_s = _probe_edit_list_offset(video_path)
        frame_offset = max(0, min(T - 1, int(round(offset_s * fps))))

    return {
        "T":            T,
        "width":        int(pose.header.dimensions.width),
        "height":       int(pose.header.dimensions.height),
        "fps":          fps,
        "frame_offset": frame_offset,   # add to round(currentTime * fps) — see edit-list note above
        "pose":         _xy(body_pts),
        "left_hand":    _xy(left_pts),
        "right_hand":   _xy(right_pts),
        "face":         _xy(face_pts),
    }


def _extract_worker(idx: int, video_path: Path, pose_path: Path) -> None:
    """Background-thread target: extract pose for an uploaded clip, then
    update its CLIPS entry in place."""
    ok = extract_pose(video_path, pose_path)
    with CLIPS_LOCK:
        CLIPS[idx]["pose_path"] = pose_path if ok else None
        CLIPS[idx]["status"] = "ready" if ok else "failed"


# ---------------------------------------------------------------------------
# Flask routes — API
# ---------------------------------------------------------------------------

@app.route("/api/clips")
def api_clips():
    out = []
    for c in CLIPS:
        entry = {"idx": c["idx"], "name": c["name"],
                  "has_pose": c["pose_path"] is not None,
                  "has_video": c["video_path"] is not None,
                  "status": c.get("status", "ready")}
        if entry["status"] == "extracting":
            entry["elapsed"] = round(time.time() - c.get("started", time.time()), 1)
        out.append(entry)
    return jsonify(out)


@app.route("/api/upload", methods=["POST"])
def api_upload():
    if UPLOAD_DIR is None:
        return jsonify({"error": "uploads not enabled"}), 400
    f = request.files.get("file")
    if f is None or not f.filename:
        return jsonify({"error": "no file in request"}), 400

    name   = Path(f.filename).name
    suffix = Path(name).suffix.lower()
    if suffix not in UPLOAD_SUFFIXES:
        return jsonify({"error": f"unsupported file type: {suffix or '(none)'}"}), 400

    dest = UPLOAD_DIR / name
    if dest.exists():
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:8]}_{name}"
    f.save(dest)

    with CLIPS_LOCK:
        idx = len(CLIPS)
        if suffix == ".pose":
            CLIPS.append({"idx": idx, "name": name, "video_path": None,
                          "pose_path": dest, "status": "ready"})
        else:
            cache_dir = POSE_CACHE_DIR or UPLOAD_DIR
            pose_path = cache_dir / (dest.name + ".pose")
            CLIPS.append({"idx": idx, "name": name, "video_path": dest,
                          "pose_path": None, "status": "extracting",
                          "started": time.time()})
        status = CLIPS[idx]["status"]

    if status == "extracting":
        threading.Thread(
            target=_extract_worker, args=(idx, dest, pose_path), daemon=True
        ).start()

    return jsonify({"idx": idx, "name": name, "status": status})


@app.errorhandler(413)
def _too_large(e):
    return jsonify({"error": "file too large"}), 413


@app.route("/api/clip/<int:idx>/activations")
def api_clip_activations(idx):
    if idx in ACT_CACHE:
        return jsonify(ACT_CACHE[idx])

    if idx < 0 or idx >= len(CLIPS):
        return jsonify({"error": "not found"}), 404
    clip = CLIPS[idx]
    if clip["pose_path"] is None:
        return jsonify({"error": "pose extraction failed for this clip"}), 400
    if MODEL is None:
        return jsonify({"error": "no model loaded"}), 400

    from stsnet.data.pose_io import load_pose_streams
    streams = load_pose_streams(clip["pose_path"], HANDEDNESS, mirror_left=True)
    if streams is None:
        return jsonify({"error": "pose load failed"}), 500

    dom    = torch.from_numpy(streams["dominant"]).unsqueeze(0).to(DEVICE)
    nondom = torch.from_numpy(streams["nondominant"]).unsqueeze(0).to(DEVICE)
    body   = torch.from_numpy(streams["body"]).unsqueeze(0).to(DEVICE)
    face   = torch.from_numpy(streams["face"]).unsqueeze(0).to(DEVICE) \
             if "face" in streams else None

    T = dom.shape[1]
    full_t = torch.tensor([T], dtype=torch.long, device=DEVICE)
    zero_t = torch.zeros(1, dtype=torch.long, device=DEVICE)

    with torch.no_grad():
        frame_feats = MODEL.frame_features(dom, nondom, body, face)  # (1, T, D)
        f = frame_feats[0]

        def _sigmoid_probs(head):
            return torch.sigmoid(head(f)).cpu().numpy()

        heads_probs = {
            "shape":  _sigmoid_probs(MODEL.shape_head),
            "att":    _sigmoid_probs(MODEL.att_head),
            "cloc":   _sigmoid_probs(MODEL.cloc_head),
            "ctype":  _sigmoid_probs(MODEL.ctype_head),
            "motion": _sigmoid_probs(MODEL.motion_head),
            "hand_type": torch.softmax(MODEL.hand_type_head(f), dim=-1).cpu().numpy(),
        }
        if MODEL.has_nondom_shape:
            heads_probs["nondom_shape"] = _sigmoid_probs(MODEL.nondom_shape_head)
        if MODEL.has_nondom_att:
            heads_probs["nondom_att"] = _sigmoid_probs(MODEL.nondom_att_head)

        # Full forward (whole-clip attention window) for the attention trace.
        out = MODEL(dom, nondom, body, face,
                     sign_start=zero_t, sign_end=full_t, lengths=full_t)
        attn = out["attn_weights"][0].cpu().numpy()  # (T,)

    def _round_list(arr):
        return [[round(float(v), 4) for v in row] for row in arr]

    def _round_1d(arr):
        return [round(float(v), 4) for v in arr]

    # Per-frame predictive entropy, in bits-per-class so every head sits on the
    # same [0, 1] scale regardless of its number of classes: binary entropy
    # (mean over classes) for the multi-hot BCE heads, categorical entropy for
    # the 2-way softmax hand_type head (whose max is also log2(2) = 1 bit).
    # Low entropy = the head is confidently committing to a (sub)set of
    # classes; high entropy = its per-frame prediction is closer to a coin
    # flip. `entropy_by_head` lets each phonological property be inspected on
    # its own; `entropy` averages across heads as a single "how confident is
    # the model right now" trace, useful as a candidate segmentation /
    # keyframe signal (see README) — note this is a diagnostic on the
    # per-frame heads themselves, which bypass the trained attention pool
    # (see predict_frames docstring), not a property the model was directly
    # supervised on.
    def _entropy(name, probs):
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        if name == "hand_type":
            return -(p * np.log2(p)).sum(axis=1)
        return (-(p * np.log2(p) + (1 - p) * np.log2(1 - p))).mean(axis=1)

    entropy_by_head = {name: _entropy(name, probs) for name, probs in heads_probs.items()}
    entropy = np.mean(list(entropy_by_head.values()), axis=0)

    label_maps = {
        "shape": VOCAB["idx_to_shape"], "att": VOCAB["idx_to_att"],
        "cloc": VOCAB["idx_to_cloc"],   "ctype": VOCAB["idx_to_ctype"],
        "motion": VOCAB["idx_to_motion"],
        "hand_type": {0: "one", 1: "two"},
        "nondom_shape": VOCAB["idx_to_shape"], "nondom_att": VOCAB["idx_to_att"],
    }

    result = {
        "T": T,
        "heads": {
            name: {
                "probs":  _round_list(probs),
                "labels": [label_maps[name].get(i, str(i)) for i in range(probs.shape[1])],
            }
            for name, probs in heads_probs.items()
        },
        "attn": _round_1d(attn),
        "entropy": _round_1d(entropy),
        "entropy_by_head": {name: _round_1d(h) for name, h in entropy_by_head.items()},
    }
    ACT_CACHE[idx] = result
    return jsonify(result)


@app.route("/api/clip/<int:idx>/keypoints")
def api_clip_keypoints(idx):
    if idx in KP_CACHE:
        return jsonify(KP_CACHE[idx])

    if idx < 0 or idx >= len(CLIPS):
        return jsonify({"error": "not found"}), 404
    clip = CLIPS[idx]
    if clip["pose_path"] is None:
        return jsonify({"error": "pose extraction failed for this clip"}), 400

    kp = _load_raw_keypoints(clip["pose_path"], clip.get("video_path"))
    if kp is None:
        return jsonify({"error": "pose load failed"}), 500
    KP_CACHE[idx] = kp
    return jsonify(kp)


# ---------------------------------------------------------------------------
# Video serving with HTTP Range support (required for <video> seeking)
# ---------------------------------------------------------------------------

@app.route("/video/<int:idx>")
def serve_video(idx):
    if idx < 0 or idx >= len(CLIPS):
        return Response("not found", 404)
    path = CLIPS[idx]["video_path"]
    if path is None or not path.exists():
        return Response("not found", 404)

    content_type = mimetypes.guess_type(str(path))[0] or "video/mp4"
    file_size    = path.stat().st_size
    range_header = request.headers.get("Range")

    if range_header:
        m     = re.match(r"bytes=(\d+)-(\d*)", range_header)
        start = int(m.group(1))
        end   = int(m.group(2)) if m.group(2) else file_size - 1
        end   = min(end, file_size - 1)
        length = end - start + 1

        def _gen_range():
            with open(path, "rb") as f:
                f.seek(start)
                remaining = length
                while remaining > 0:
                    chunk = f.read(min(65536, remaining))
                    if not chunk:
                        break
                    remaining -= len(chunk)
                    yield chunk

        return Response(
            _gen_range(), 206,
            headers={
                "Content-Range":  f"bytes {start}-{end}/{file_size}",
                "Accept-Ranges":  "bytes",
                "Content-Length": str(length),
                "Content-Type":   content_type,
            },
        )

    def _gen_full():
        with open(path, "rb") as f:
            while True:
                chunk = f.read(65536)
                if not chunk:
                    break
                yield chunk

    return Response(
        _gen_full(),
        headers={
            "Content-Length": str(file_size),
            "Accept-Ranges":  "bytes",
            "Content-Type":   content_type,
        },
    )


# ---------------------------------------------------------------------------
# Frontend
# ---------------------------------------------------------------------------

HEAD_TITLES = {
    "shape": "Shape", "att": "Attitude", "cloc": "Contact Location",
    "ctype": "Contact Type", "motion": "Motion", "hand_type": "Hand Type",
    "nondom_shape": "Shape (non-dom.)", "nondom_att": "Attitude (non-dom.)",
}
HEAD_ORDER = ["shape", "att", "cloc", "ctype", "motion", "hand_type",
              "nondom_shape", "nondom_att"]
N_SHOW = {"shape": 14, "att": 10, "cloc": 14, "ctype": 4, "motion": 8,
          "hand_type": 2, "nondom_shape": 14, "nondom_att": 10}

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>STS-Net Inspector</title>
<style>
* { box-sizing: border-box; margin: 0; padding: 0; }
body {
  font-family: 'Segoe UI', system-ui, sans-serif;
  background: #1a1a2e;
  color: #e8e8e8;
  height: 100vh;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}

/* ── header ── */
#header {
  padding: 8px 12px;
  background: #16213e;
  border-bottom: 1px solid #0f3460;
  display: flex;
  align-items: center;
  gap: 10px;
  flex-shrink: 0;
}
#header h1 { font-size: 15px; font-weight: 600; color: #e94560; flex-shrink: 0; }
#search {
  flex: 1;
  background: #0f3460;
  border: 1px solid #1a4080;
  color: #e8e8e8;
  padding: 5px 8px;
  border-radius: 4px;
  font-size: 13px;
}
#search::placeholder { color: #888; }
#count { font-size: 11px; color: #888; flex-shrink: 0; }

/* ── main layout ── */
#main { display: flex; flex: 1; min-height: 0; position: relative; }

/* ── sidebar ── */
#sidebar { width: 260px; flex-shrink: 0; overflow-y: auto; border-right: 1px solid #0f3460; }
.clip-item { padding: 7px 10px; cursor: pointer; border-bottom: 1px solid #0f3460; font-size: 13px; word-break: break-all; }
.clip-item:hover { background: #16213e; }
.clip-item.active { background: #0f3460; border-left: 3px solid #e94560; }
.clip-item.no-pose { color: #666; cursor: not-allowed; }
.clip-item.status-extracting { color: #f0c060; cursor: default; }
.clip-item.status-failed { color: #e94560; cursor: default; }
.clip-item.uploading { color: #9ab; cursor: default; }

/* ── upload progress ── */
.upload-progress-wrap { margin-top: 4px; }
.upload-status-text { font-size: 10px; color: #999; margin-bottom: 3px; }
.upload-progress-bar-bg { height: 4px; border-radius: 2px; background: #0f3460; overflow: hidden; }
.upload-progress-bar { height: 100%; width: 0%; background: linear-gradient(90deg,#e94560,#f0c060); border-radius: 2px; transition: width 0.2s ease; }
.upload-progress-bar.indeterminate {
  width: 35% !important;
  animation: indet 1.1s ease-in-out infinite;
}
@keyframes indet {
  0%   { margin-left: -35%; }
  100% { margin-left: 100%; }
}

/* ── drag-and-drop overlay ── */
#dropHint {
  position: absolute; inset: 0; z-index: 20;
  display: none; align-items: center; justify-content: center;
  background: rgba(15,33,62,0.92);
  border: 3px dashed #e94560;
  color: #e94560;
  font-size: 16px; font-weight: 600;
  pointer-events: none;
}
#dropHint.show { display: flex; }

/* ── viewer ── */
#viewer {
  flex: 1;
  display: flex;
  flex-direction: column;
  min-width: 0;
  padding: 8px 10px;
  gap: 6px;
  overflow-y: auto;
}
#videoWrap {
  flex-shrink: 0;
  background: #000;
  border-radius: 4px;
  overflow: hidden;
  display: flex;
  flex-direction: column;
  max-height: 42vh;
}
#videoToolbar {
  flex-shrink: 0;
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 4px 10px;
  background: #0d1428;
  font-size: 11px;
  color: #aab4cc;
}
#videoToolbar label { display: flex; align-items: center; gap: 5px; cursor: pointer; user-select: none; }
#videoToolbar input[type=checkbox] { cursor: pointer; }
#kpLegend { display: flex; align-items: center; gap: 10px; margin-left: auto; }
#kpLegend span { display: flex; align-items: center; gap: 4px; }
#kpLegend .sw { width: 8px; height: 8px; border-radius: 50%; display: inline-block; }
#videoStage { position: relative; display: flex; justify-content: center; overflow: hidden; flex: 1; min-height: 0; }
#videoBox { position: relative; display: inline-block; line-height: 0; }
video { max-height: 38vh; max-width: 100%; display: block; }
#kpCanvas {
  position: absolute;
  top: 0; left: 0;
  width: 100%; height: 100%;
  pointer-events: none;
  display: none;
}
#kpCanvas.show { display: block; }

#empty { display: flex; align-items: center; justify-content: center; flex: 1; color: #555; font-size: 14px; }

/* ── activation sections ── */
.act-section { flex-shrink: 0; border: 1px solid #1a3060; border-radius: 5px; overflow: hidden; }
.act-header {
  display: flex;
  align-items: center;
  gap: 6px;
  padding: 5px 10px;
  background: #16213e;
  cursor: pointer;
  user-select: none;
  font-size: 12px;
  font-weight: 600;
  color: #c0d0f0;
  border-bottom: 1px solid #1a3060;
}
.act-header:hover { background: #1a2a4a; }
.act-arrow { font-size: 10px; transition: transform 0.15s; }
.act-section.collapsed .act-arrow { transform: rotate(-90deg); }
.act-body {
  max-height: 600px;
  overflow: hidden;
  transition: max-height 0.2s ease;
  background: #12192e;
  padding: 6px 8px;
}
.act-section.collapsed .act-body { max-height: 0; padding: 0 8px; }
.act-loading { font-size: 11px; color: #888; padding: 8px 0; }
.canvas-wrap { width: 100%; }
canvas.act-canvas { width: 100%; display: block; }
</style>
</head>
<body>

<div id="header">
  <h1>STS-Net Inspector</h1>
  <input id="search" type="text" placeholder="filter filenames…" autocomplete="off">
  <span id="count"></span>
</div>

<div id="main">
  <div id="dropHint">Drop video files to add clips</div>
  <div id="sidebar"><div id="clipList"></div></div>

  <div id="viewer">
    <div id="empty">← select a clip</div>

    <div id="videoWrap" style="display:none">
      <div id="videoToolbar">
        <label><input type="checkbox" id="kpToggle"> Show keypoints</label>
        <div id="kpLegend">
          <span><i class="sw" style="background:#4fd1c5"></i>pose</span>
          <span><i class="sw" style="background:#e94560"></i>left hand</span>
          <span><i class="sw" style="background:#5aa9ff"></i>right hand</span>
          <span><i class="sw" style="background:#f0c060"></i>face</span>
        </div>
      </div>
      <div id="videoStage">
        <div id="videoBox">
          <video id="video" controls preload="metadata"></video>
          <canvas id="kpCanvas"></canvas>
        </div>
      </div>
    </div>

    <div id="sections"></div>
  </div><!-- #viewer -->
</div><!-- #main -->

<script>
// ── constants ──────────────────────────────────────────────────────────────
const LABEL_W = 90;   // px for row label column
const AXIS_H  = 18;   // px for time axis
const ROW_H   = 14;   // px per class row in heatmap
const N_SHOW  = __N_SHOW__;
const HEAD_ORDER  = __HEAD_ORDER__;
const HEAD_TITLES = __HEAD_TITLES__;

// Standard MediaPipe skeleton topologies, for the keypoint overlay.
const POSE_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,7],[0,4],[4,5],[5,6],[6,8],[9,10],
  [11,12],[11,13],[13,15],[15,17],[15,19],[15,21],[17,19],
  [12,14],[14,16],[16,18],[16,20],[16,22],[18,20],
  [11,23],[12,24],[23,24],[23,25],[24,26],[25,27],[26,28],
  [27,29],[28,30],[27,31],[28,32],[29,31],[30,32],
];
const HAND_CONNECTIONS = [
  [0,1],[1,2],[2,3],[3,4],
  [0,5],[5,6],[6,7],[7,8],
  [5,9],[9,10],[10,11],[11,12],
  [9,13],[13,14],[14,15],[15,16],
  [13,17],[17,18],[18,19],[19,20],
  [0,17],
];

// ── state ──────────────────────────────────────────────────────────────────
let clips      = [];
let activeIdx  = null;
let actData    = null;   // current activations JSON
let headKeys   = [];     // heads present in the current actData, in display order
const collapseState = {};
const uploads  = {};     // id -> {name, pct, error}  (in-flight browser→server uploads)
const ALLOWED_UPLOAD_SUFFIXES = ['.mp4', '.mov', '.avi', '.mkv', '.webm', '.pose'];
let pollTimer  = null;   // polls /api/clips while any clip is still extracting
let kpData     = null;   // current clip's raw keypoint overlay JSON
let kpEnabled  = false;  // "Show keypoints" toggle state (persists across clips)

// ── YlOrRd colormap ────────────────────────────────────────────────────────
function ylOrRd(t) {
  const stops = [
    [1.0,  0.988, 0.910],  // 0.0
    [0.996, 0.769, 0.310], // 0.25
    [0.988, 0.553, 0.349], // 0.5
    [0.890, 0.290, 0.200], // 0.75
    [0.545, 0.0,   0.0  ], // 1.0
  ];
  const scaled = Math.min(1, Math.max(0, t)) * 4;
  const lo = Math.floor(scaled), hi = Math.min(4, lo + 1);
  const f  = scaled - lo;
  const r = stops[lo][0] + f * (stops[hi][0] - stops[lo][0]);
  const g = stops[lo][1] + f * (stops[hi][1] - stops[lo][1]);
  const b = stops[lo][2] + f * (stops[hi][2] - stops[lo][2]);
  return `rgb(${Math.round(r*255)},${Math.round(g*255)},${Math.round(b*255)})`;
}

// ── collapsible sections ───────────────────────────────────────────────────
function toggleSection(key) {
  collapseState[key] = !collapseState[key];
  document.getElementById('sec-' + key).classList.toggle('collapsed', collapseState[key]);
  if (!collapseState[key] && actData) {
    if (key === 'attn') redrawAttn();
    else if (key === 'entropy') redrawEntropy();
    else redrawHeatmap(key);
  }
}

// ── clip list ──────────────────────────────────────────────────────────────
async function fetchClips() {
  const r = await fetch('/api/clips');
  clips = await r.json();
  renderList();
}

function renderList() {
  const q  = document.getElementById('search').value.trim().toLowerCase();
  const cl = document.getElementById('clipList');
  const shown = clips.filter(c => !q || c.name.toLowerCase().includes(q));
  document.getElementById('count').textContent = shown.length + ' / ' + clips.length + ' clips';
  cl.innerHTML = '';

  // In-flight browser→server uploads (not yet acknowledged by the server).
  Object.values(uploads).forEach(u => {
    const el = document.createElement('div');
    el.className = 'clip-item uploading';
    const statusText = u.error ? ('Error: ' + u.error) : ('Uploading… ' + u.pct + '%');
    el.innerHTML = `
      <div>${u.name}</div>
      <div class="upload-progress-wrap">
        <div class="upload-status-text"${u.error ? ' style="color:#e94560"' : ''}>${statusText}</div>
        <div class="upload-progress-bar-bg"><div class="upload-progress-bar" style="width:${u.pct}%"></div></div>
      </div>`;
    cl.appendChild(el);
  });

  shown.forEach(c => {
    const status    = c.status || 'ready';
    const clickable = status === 'ready' && c.has_pose;
    const el = document.createElement('div');
    el.className = 'clip-item'
      + (activeIdx === c.idx ? ' active' : '')
      + (clickable ? '' : ' no-pose')
      + (status === 'extracting' ? ' status-extracting' : '')
      + (status === 'failed'     ? ' status-failed'     : '');

    if (status === 'extracting') {
      const elapsed = c.elapsed != null ? Math.round(c.elapsed) + 's' : '';
      el.innerHTML = `
        <div>${c.name}</div>
        <div class="upload-progress-wrap">
          <div class="upload-status-text">Extracting pose… ${elapsed}</div>
          <div class="upload-progress-bar-bg"><div class="upload-progress-bar indeterminate"></div></div>
        </div>`;
    } else if (status === 'failed') {
      el.textContent = c.name + '  (pose extraction failed)';
    } else {
      el.textContent = c.name + (c.has_pose ? '' : '  (pose extraction failed)');
    }
    el.onclick = () => { if (clickable) selectClip(c); };
    cl.appendChild(el);
  });
}

document.getElementById('search').addEventListener('input', renderList);

// ── drag-and-drop upload ────────────────────────────────────────────────────
function ensurePolling() {
  if (pollTimer) return;
  pollTimer = setInterval(async () => {
    await fetchClips();
    if (!clips.some(c => c.status === 'extracting')) {
      clearInterval(pollTimer);
      pollTimer = null;
    }
  }, 1200);
}

function uploadFile(file) {
  const suffix = '.' + (file.name.split('.').pop() || '').toLowerCase();
  if (!ALLOWED_UPLOAD_SUFFIXES.includes(suffix)) {
    alert('Unsupported file type: ' + file.name);
    return;
  }

  const id = 'u' + Math.random().toString(36).slice(2);
  uploads[id] = { name: file.name, pct: 0 };
  renderList();

  const xhr = new XMLHttpRequest();
  xhr.open('POST', '/api/upload');
  xhr.upload.addEventListener('progress', e => {
    if (e.lengthComputable) {
      uploads[id].pct = Math.round((e.loaded / e.total) * 100);
      renderList();
    }
  });
  xhr.onload = async () => {
    if (xhr.status !== 200) {
      let msg = 'upload failed';
      try { msg = JSON.parse(xhr.responseText).error || msg; } catch (err) {}
      uploads[id].error = msg;
      renderList();
      setTimeout(() => { delete uploads[id]; renderList(); }, 5000);
      return;
    }
    delete uploads[id];
    await fetchClips();
    ensurePolling();
  };
  xhr.onerror = () => {
    uploads[id].error = 'network error';
    renderList();
    setTimeout(() => { delete uploads[id]; renderList(); }, 5000);
  };

  const form = new FormData();
  form.append('file', file);
  xhr.send(form);
}

const mainEl   = document.getElementById('main');
const dropHint = document.getElementById('dropHint');
let dragDepth  = 0;

// Prevent the browser from navigating to the dropped file anywhere on the page.
window.addEventListener('dragover', e => e.preventDefault());
window.addEventListener('drop',     e => e.preventDefault());

mainEl.addEventListener('dragenter', e => {
  e.preventDefault();
  dragDepth++;
  dropHint.classList.add('show');
});
mainEl.addEventListener('dragover', e => e.preventDefault());
mainEl.addEventListener('dragleave', e => {
  e.preventDefault();
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) dropHint.classList.remove('show');
});
mainEl.addEventListener('drop', e => {
  e.preventDefault();
  dragDepth = 0;
  dropHint.classList.remove('show');
  const files = Array.from(e.dataTransfer.files || []);
  files.forEach(uploadFile);
});

// ── video element ──────────────────────────────────────────────────────────
const video    = document.getElementById('video');
const kpCanvas = document.getElementById('kpCanvas');
const kpToggle = document.getElementById('kpToggle');
video.addEventListener('timeupdate', () => { if (actData) redrawAll(); });

// ── keypoint overlay ─────────────────────────────────────────────────────────
async function ensureKeypoints() {
  if (!kpEnabled || activeIdx == null) return;
  if (kpData && kpData._idx === activeIdx) return;
  kpData = null;
  try {
    const r = await fetch('/api/clip/' + activeIdx + '/keypoints');
    if (!r.ok) return;
    const data = await r.json();
    data._idx = activeIdx;
    kpData = data;
  } catch (e) { kpData = null; }
}

function drawKeypoints() {
  const ctx = kpCanvas.getContext('2d');
  if (kpCanvas.width === 0 || kpCanvas.height === 0) return;
  ctx.clearRect(0, 0, kpCanvas.width, kpCanvas.height);
  if (!kpEnabled || !kpData || !kpData.fps) return;

  // Use the pose file's own fps + edit-list frame offset rather than
  // deriving fps from video.duration — a clip trimmed mid-stream with
  // `-c copy` can have leading frames hidden from playback by an MP4 edit
  // list that video_to_pose still decodes, which would otherwise desync
  // the overlay near the start of the clip (see _probe_edit_list_offset
  // server-side). Frame 0 in kpData already accounts for that offset.
  const f = Math.max(0, Math.min(
    kpData.T - 1,
    Math.round(video.currentTime * kpData.fps) + (kpData.frame_offset || 0)
  ));

  const drawPts = (pts, color, connections, radius) => {
    const frame = pts && pts[f];
    if (!frame) return;
    if (connections) {
      ctx.strokeStyle = color;
      ctx.lineWidth   = Math.max(1, kpCanvas.width / 400);
      connections.forEach(([a, b]) => {
        const pa = frame[a], pb = frame[b];
        if (!pa || !pb) return;
        ctx.beginPath(); ctx.moveTo(pa[0], pa[1]); ctx.lineTo(pb[0], pb[1]); ctx.stroke();
      });
    }
    ctx.fillStyle = color;
    frame.forEach(p => {
      if (!p) return;
      ctx.beginPath(); ctx.arc(p[0], p[1], radius, 0, 2 * Math.PI); ctx.fill();
    });
  };

  drawPts(kpData.pose,       '#4fd1c5', POSE_CONNECTIONS, kpCanvas.width / 220);
  drawPts(kpData.face,       '#f0c060', null,              kpCanvas.width / 350);
  drawPts(kpData.left_hand,  '#e94560', HAND_CONNECTIONS,  kpCanvas.width / 300);
  drawPts(kpData.right_hand, '#5aa9ff', HAND_CONNECTIONS,  kpCanvas.width / 300);
}

function kpLoop() {
  if (video.paused || video.ended) return;
  drawKeypoints();
  requestAnimationFrame(kpLoop);
}

video.addEventListener('loadedmetadata', () => {
  kpCanvas.width  = video.videoWidth;
  kpCanvas.height = video.videoHeight;
  drawKeypoints();
});
video.addEventListener('play',    () => requestAnimationFrame(kpLoop));
video.addEventListener('seeked',  drawKeypoints);
video.addEventListener('pause',   drawKeypoints);

kpToggle.addEventListener('change', async () => {
  kpEnabled = kpToggle.checked;
  kpCanvas.classList.toggle('show', kpEnabled);
  if (kpEnabled) await ensureKeypoints();
  drawKeypoints();
});

// ── sections DOM (built dynamically from the heads present) ────────────────
function buildSections(keys) {
  const wrap = document.getElementById('sections');
  wrap.innerHTML = '';
  keys.concat(['attn', 'entropy']).forEach(key => {
    if (!(key in collapseState)) collapseState[key] = false;
    const title = key === 'attn' ? 'Attention'
                : key === 'entropy' ? 'Entropy (bits/class, mean over heads)'
                : (HEAD_TITLES[key] || key);
    const sec = document.createElement('div');
    sec.className = 'act-section' + (collapseState[key] ? ' collapsed' : '');
    sec.id = 'sec-' + key;
    sec.innerHTML = `
      <div class="act-header" onclick="toggleSection('${key}')">
        <span class="act-arrow">▾</span> ${title}
      </div>
      <div class="act-body">
        <div class="act-loading" id="loading-${key}">Loading…</div>
        <div class="canvas-wrap" id="wrap-${key}" style="display:none"><canvas class="act-canvas" id="canvas-${key}"></canvas></div>
      </div>`;
    wrap.appendChild(sec);
  });
}

// ── clip selection ─────────────────────────────────────────────────────────
async function selectClip(c) {
  activeIdx = c.idx;
  actData   = null;
  kpData    = null;
  renderList();

  document.getElementById('empty').style.display    = 'none';
  document.getElementById('videoWrap').style.display = c.has_video ? 'flex' : 'none';

  if (c.has_video) {
    video.src = '/video/' + c.idx;
    video.load();
    if (kpEnabled) await ensureKeypoints();
  } else {
    video.removeAttribute('src');
  }

  // Placeholder sections until we know which heads this checkpoint has.
  buildSections(headKeys.length ? headKeys : HEAD_ORDER);

  try {
    const ra = await fetch('/api/clip/' + c.idx + '/activations');
    if (!ra.ok) {
      const err = await ra.json();
      document.querySelectorAll('.act-loading').forEach(el => el.textContent = 'Error: ' + (err.error || ra.status));
      return;
    }
    actData  = await ra.json();
    headKeys = HEAD_ORDER.filter(k => k in actData.heads);
    buildSections(headKeys);
    headKeys.concat(['attn', 'entropy']).forEach(key => {
      document.getElementById('loading-' + key).style.display = 'none';
      document.getElementById('wrap-' + key).style.display = 'block';
    });
    redrawAll();
  } catch (e) {
    document.querySelectorAll('.act-loading').forEach(el => el.textContent = 'Error: ' + e.message);
  }
}

// ── drawing helpers ────────────────────────────────────────────────────────

function drawTimeAxis(ctx, W, H, axisY, T) {
  ctx.fillStyle = '#0a1a32';
  ctx.fillRect(LABEL_W, axisY, W - LABEL_W, AXIS_H);

  const plotW = W - LABEL_W;
  const dur   = T;

  const candidates = [1, 2, 5, 10, 20, 50, 100, 200];
  let interval = candidates[candidates.length - 1];
  for (const iv of candidates) {
    if (iv / dur * plotW >= 40) { interval = iv; break; }
  }

  ctx.textAlign    = 'center';
  ctx.textBaseline = 'middle';
  for (let t = 0; t <= dur; t += interval) {
    const x = LABEL_W + t / dur * plotW;
    ctx.strokeStyle = 'rgba(255,255,255,0.06)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, axisY); ctx.stroke();
    ctx.strokeStyle = 'rgba(255,255,255,0.35)';
    ctx.lineWidth   = 1;
    ctx.beginPath(); ctx.moveTo(x, axisY); ctx.lineTo(x, axisY + 5); ctx.stroke();
    ctx.fillStyle = 'rgba(200,200,200,0.6)';
    ctx.font = '9px system-ui';
    ctx.fillText(String(t), x, axisY + AXIS_H / 2 + 1);
  }
}

function drawPlayhead(ctx, H, axisY, W) {
  if (!video.duration) return;
  const plotW = W - LABEL_W;
  const x = LABEL_W + (video.currentTime / video.duration) * plotW;
  ctx.strokeStyle = 'rgba(233,69,96,0.4)';
  ctx.lineWidth   = 1;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
  ctx.strokeStyle = '#e94560';
  ctx.lineWidth   = 1.5;
  ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, axisY); ctx.stroke();
  ctx.fillStyle = '#e94560';
  ctx.beginPath();
  ctx.moveTo(x - 4, 0); ctx.lineTo(x + 4, 0); ctx.lineTo(x, 6);
  ctx.closePath(); ctx.fill();
}

// Pick which class rows to show: highest max-activation first, up to nShow.
function pickRows(probs, nShow, pinFirst) {
  const T = probs.length;
  const C = probs[0].length;
  const maxAct = new Array(C).fill(0);
  for (let t = 0; t < T; t++)
    for (let c = 0; c < C; c++)
      if (probs[t][c] > maxAct[c]) maxAct[c] = probs[t][c];

  const pinned = pinFirst != null ? [pinFirst] : [];
  const others = [];
  for (let c = 0; c < C; c++)
    if (c !== pinFirst) others.push(c);
  others.sort((a, b) => maxAct[b] - maxAct[a]);

  return [...pinned, ...others].slice(0, nShow);
}

function drawHeatmap(canvasEl, headData, T) {
  const headKey = canvasEl.id.replace('canvas-', '');
  const nShow   = N_SHOW[headKey] || 14;
  const probs   = headData.probs;   // (T, C)
  const labels  = headData.labels;  // C

  // pin "none" (index 0) as first row for cloc/ctype/motion so the
  // no-activity case is always visible.
  const pinFirst = (headKey === 'motion' || headKey === 'cloc' || headKey === 'ctype') ? 0 : null;
  const rows = pickRows(probs, nShow, pinFirst);

  const plotH = rows.length * ROW_H;
  const totalH = plotH + AXIS_H;
  const W = canvasEl.width = canvasEl.offsetWidth;
  canvasEl.height = totalH;

  const ctx   = canvasEl.getContext('2d');
  const plotW = W - LABEL_W;
  ctx.clearRect(0, 0, W, totalH);

  const cellW = plotW / T;
  rows.forEach((ci, ri) => {
    const y0 = ri * ROW_H;
    for (let t = 0; t < T; t++) {
      ctx.fillStyle = ylOrRd(probs[t][ci]);
      ctx.fillRect(LABEL_W + t * cellW, y0, Math.ceil(cellW), ROW_H);
    }
    ctx.fillStyle    = 'rgba(200,200,200,0.45)';
    ctx.font         = '9px system-ui';
    ctx.textAlign    = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(labels[ci] || String(ci), LABEL_W - 4, y0 + ROW_H / 2);
  });

  drawTimeAxis(ctx, W, totalH, plotH, T);
  drawPlayhead(ctx, totalH, plotH, W);
}

// Generic filled line-plot for a single 1-D per-frame trace (attention
// weights, entropy, ...). `fixedMax`, if given, pins the y-scale (e.g. 1.0
// for a bits-per-class entropy trace, whose theoretical max is known) so it
// reads consistently across clips instead of auto-scaling to each clip's own
// peak.
function drawLine1D(canvasEl, data, T, label, fixedMax, color) {
  const PLOT_H = 50;
  const totalH = PLOT_H + AXIS_H;
  const W      = canvasEl.width = canvasEl.offsetWidth;
  canvasEl.height = totalH;

  const ctx   = canvasEl.getContext('2d');
  const plotW = W - LABEL_W;
  ctx.clearRect(0, 0, W, totalH);

  ctx.fillStyle = '#0d1a30';
  ctx.fillRect(LABEL_W, 0, plotW, PLOT_H);

  ctx.fillStyle    = 'rgba(200,200,200,0.45)';
  ctx.font         = '9px system-ui';
  ctx.textAlign    = 'right';
  ctx.textBaseline = 'middle';
  ctx.fillText(label, LABEL_W - 4, PLOT_H / 2);

  const maxVal = fixedMax != null ? fixedMax : Math.max(...data, 1e-9);
  const norm   = data.map(v => Math.min(1, v / maxVal));

  const cellW = plotW / T;
  ctx.fillStyle = color || 'steelblue';
  ctx.beginPath();
  ctx.moveTo(LABEL_W, PLOT_H);
  for (let t = 0; t < T; t++) {
    const x = LABEL_W + (t + 0.5) * cellW;
    const y = PLOT_H * (1 - norm[t]);
    if (t === 0) ctx.moveTo(LABEL_W, PLOT_H);
    ctx.lineTo(x, y);
  }
  ctx.lineTo(LABEL_W + plotW, PLOT_H);
  ctx.closePath();
  ctx.fill();

  drawTimeAxis(ctx, W, totalH, PLOT_H, T);
  drawPlayhead(ctx, totalH, PLOT_H, W);
}

function drawAttn(canvasEl, attn, T) {
  drawLine1D(canvasEl, attn, T, 'attn');
}

function drawEntropy(canvasEl, entropy, T) {
  drawLine1D(canvasEl, entropy, T, 'H (bits)', 1.0, '#e9a544');
}

// ── redraw functions (check collapsed state) ──
function redrawHeatmap(key) {
  if (collapseState[key] || !actData) return;
  const canvas = document.getElementById('canvas-' + key);
  const head   = actData.heads[key];
  if (!canvas || !head) return;
  drawHeatmap(canvas, head, actData.T);
}

function redrawAttn() {
  if (collapseState['attn'] || !actData) return;
  const canvas = document.getElementById('canvas-attn');
  if (!canvas) return;
  drawAttn(canvas, actData.attn, actData.T);
}

function redrawEntropy() {
  if (collapseState['entropy'] || !actData) return;
  const canvas = document.getElementById('canvas-entropy');
  if (!canvas) return;
  drawEntropy(canvas, actData.entropy, actData.T);
}

function redrawAll() {
  headKeys.forEach(k => redrawHeatmap(k));
  redrawAttn();
  redrawEntropy();
}

// Redraw the active section's canvas when its container is resized.
new ResizeObserver(() => { if (actData) redrawAll(); }).observe(document.getElementById('sections'));

// ── init ───────────────────────────────────────────────────────────────────
fetchClips();
</script>
</body>
</html>
"""

HTML = (HTML
        .replace("__N_SHOW__", str(N_SHOW).replace("'", '"'))
        .replace("__HEAD_ORDER__", str(HEAD_ORDER).replace("'", '"'))
        .replace("__HEAD_TITLES__", str(HEAD_TITLES).replace("'", '"')))


@app.route("/")
def index():
    return render_template_string(HTML)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    global CLIPS, MODEL, VOCAB, DEVICE, HANDEDNESS, UPLOAD_DIR, POSE_CACHE_DIR

    ap = argparse.ArgumentParser(
        description="Interactive STS-Net v0.2 activation inspector")
    ap.add_argument("videos", nargs="*", default=[],
                    help="Video (.mp4, .mov, ...) or .pose files to inspect "
                         "(more can be added later by dragging them into the browser)")
    ap.add_argument("--ckpt", default="checkpoints/stsnet_v02.pt",
                    help="ClipClassifier checkpoint (default: checkpoints/stsnet_v02.pt)")
    ap.add_argument("--handedness", default="right", choices=["right", "left"])
    ap.add_argument("--device", default="cpu",
                    help="Torch device (default: cpu; use cuda for GPU)")
    ap.add_argument("--pose_cache_dir", default=None,
                    help="Directory to cache extracted .pose files "
                         "(default: alongside each video)")
    ap.add_argument("--upload_dir", default=None,
                    help="Directory to save files dragged into the browser "
                         "(default: a temp directory)")
    ap.add_argument("--no_upload", action="store_true",
                    help="Disable the browser drag-and-drop upload feature")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--host", default="0.0.0.0")
    args = ap.parse_args()

    HANDEDNESS = args.handedness
    POSE_CACHE_DIR = Path(args.pose_cache_dir) if args.pose_cache_dir else None
    if POSE_CACHE_DIR:
        POSE_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if not args.no_upload:
        UPLOAD_DIR = Path(args.upload_dir) if args.upload_dir \
            else Path(tempfile.mkdtemp(prefix="stsnet_inspector_uploads_"))
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Drag-and-drop uploads enabled, saving to: {UPLOAD_DIR}")

    # ── 1. Resolve videos → pose files (extracting where needed) ────────────
    print(f"Preparing {len(args.videos)} clip(s)…")
    for i, raw in enumerate(args.videos):
        path = Path(raw)
        if not path.exists():
            print(f"  WARNING: not found, skipping: {path}")
            continue

        if path.suffix.lower() == ".pose":
            pose_path, video_path = path, None
        elif path.suffix.lower() in VIDEO_SUFFIXES:
            video_path = path
            cache_dir  = POSE_CACHE_DIR or path.parent
            pose_path  = cache_dir / (path.name + ".pose")
            if not pose_path.exists():
                ok = extract_pose(path, pose_path)
                pose_path = pose_path if ok else None
        else:
            print(f"  WARNING: unsupported file type, skipping: {path}")
            continue

        CLIPS.append({
            "idx": len(CLIPS), "name": path.name,
            "video_path": video_path, "pose_path": pose_path,
            "status": "ready" if pose_path is not None else "failed",
        })

    n_ok = sum(1 for c in CLIPS if c["pose_path"] is not None)
    print(f"  {n_ok}/{len(CLIPS)} clips ready")

    # ── 2. Load ClipClassifier checkpoint ────────────────────────────────────
    DEVICE = torch.device(
        args.device if (args.device != "cuda" or torch.cuda.is_available()) else "cpu"
    )
    print(f"Loading ClipClassifier from {args.ckpt} on {DEVICE}…")
    from stsnet.clip_classifier import ClipClassifier
    MODEL, vocab_meta = ClipClassifier.from_checkpoint(args.ckpt, map_location=str(DEVICE))
    MODEL.to(DEVICE)
    MODEL.eval()

    def _invert(d):
        return {v: k for k, v in d.items()}

    VOCAB["idx_to_shape"]  = _invert(vocab_meta.get("shape_to_idx",  {}))
    VOCAB["idx_to_att"]    = _invert(vocab_meta.get("att_to_idx",    {}))
    VOCAB["idx_to_motion"] = _invert(vocab_meta.get("motion_to_idx", {}))
    VOCAB["idx_to_cloc"]   = _invert(vocab_meta.get("cloc_to_idx",   {}))
    VOCAB["idx_to_ctype"]  = _invert(vocab_meta.get("ctype_to_idx",  {}))

    print(f"\nStarting Inspector on http://{args.host}:{args.port}/")
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == "__main__":
    main()
