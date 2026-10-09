"""
STS-Net demo — Gradio app for Hugging Face Spaces.

Upload a short sign-language video; the app extracts MediaPipe Holistic
pose, runs it through STS-Net v0.2 (`stsnet_v02_mas.pt`, the monotonic-
alignment checkpoint — ordered, temporally coherent per-frame predictions
rather than the flickering per-frame output of the attention-pooled
checkpoint), and shows the same per-frame activation heatmaps as
scripts/inspector.py, one collapsible pane per head, plus a MediaPipe
keypoint overlay on the video.

Deployed from the `demo/` directory of github.com/jbeskow/stsnet — see
demo/README.md for the exact file layout uploaded to the Space.

Runs on a free ZeroGPU Space: the model forward pass (the only real torch
compute — pose extraction is a CPU subprocess) is wrapped in `_run_model`,
decorated with `@spaces.GPU` so it gets a real GPU attached for that call.
`spaces.GPU` is a no-op outside the ZeroGPU runtime, so this also runs
unmodified on CPU (or a local GPU) for development.
"""

import base64
import html
import os
import subprocess
import tempfile
import shutil
from io import BytesIO
from pathlib import Path

import cv2
import gradio as gr
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
import spaces
import torch

from stsnet.clip_classifier import ClipClassifier
from stsnet.data.pose_io import load_pose_streams

import scripts.inspector as inspector
from scripts.inspector import (
    extract_pose,
    _model_activations,
    _load_raw_keypoints,
    HEAD_TITLES,
    HEAD_ORDER,
    N_SHOW,
)

# ---------------------------------------------------------------------------
# Model (loaded once at startup)
# ---------------------------------------------------------------------------

CKPT_PATH      = os.environ.get("STSNET_CKPT", "checkpoints/stsnet_v02_mas.pt")
HANDEDNESS     = "right"
MAX_DURATION_S = float(os.environ.get("STSNET_MAX_DURATION", 20))
DEVICE         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
VIDEO_ELEM_ID  = "stsnet_video"
inspector.DEVICE = DEVICE   # _model_activations reads this module-level global


def _invert(d: dict) -> dict:
    return {v: k for k, v in d.items()}


print(f"Loading ClipClassifier from {CKPT_PATH}...")
_model, _vocab_meta = ClipClassifier.from_checkpoint(CKPT_PATH, map_location=str(DEVICE))
_model.to(DEVICE)
_model.eval()

_ckpt_n_dims = _vocab_meta.get("model_kwargs", {}).get("n_dims", 3)
_objective   = _vocab_meta.get("objective", "ap")

MODEL_ENTRY = {
    "name": Path(CKPT_PATH).stem,
    "model": _model,
    "no_z": _ckpt_n_dims == 2,
    "per_frame": "softmax" if _objective.startswith("mas") else "sigmoid",
    "vocab": {
        "idx_to_shape":  _invert(_vocab_meta.get("shape_to_idx",  {})),
        "idx_to_att":    _invert(_vocab_meta.get("att_to_idx",    {})),
        "idx_to_motion": _invert(_vocab_meta.get("motion_to_idx", {})),
        "idx_to_cloc":   _invert(_vocab_meta.get("cloc_to_idx",   {})),
        "idx_to_ctype":  _invert(_vocab_meta.get("ctype_to_idx",  {})),
    },
}
print(f"  input: {'2D (xy only)' if MODEL_ENTRY['no_z'] else '3D (xyz)'}  "
      f"objective: {_objective}  per-frame: {MODEL_ENTRY['per_frame']}")

# ---------------------------------------------------------------------------
# Keypoint overlay drawing (same topology/colors as the inspector's JS)
# ---------------------------------------------------------------------------

_POSE_CONN = mp.solutions.pose.POSE_CONNECTIONS
_HAND_CONN = mp.solutions.hands.HAND_CONNECTIONS

# Matches scripts/inspector.py's drawKeypoints() stream colors (hex -> BGR).
_STREAM_STYLE = {
    "pose":       ((197, 209, 79),  _POSE_CONN, 3),
    "face":       ((96, 192, 240),  None,       1),
    "left_hand":  ((96, 69, 233),   _HAND_CONN, 2),
    "right_hand": ((255, 169, 90),  _HAND_CONN, 2),
}


def _draw_overlay(frame: np.ndarray, kp: dict, t: int) -> np.ndarray:
    for stream, (color, connections, radius) in _STREAM_STYLE.items():
        pts = kp[stream][t]
        if connections:
            for a, b in connections:
                pa, pb = pts[a], pts[b]
                if pa is not None and pb is not None:
                    cv2.line(frame, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                              color, 1, cv2.LINE_AA)
        for p in pts:
            if p is not None:
                cv2.circle(frame, (int(p[0]), int(p[1])), radius, color, -1, cv2.LINE_AA)
    return frame


def _build_overlay_video(src_video: Path, kp: dict, out_path: Path, progress=None) -> None:
    cap = cv2.VideoCapture(str(src_video))
    fps    = cap.get(cv2.CAP_PROP_FPS) or kp["fps"]
    width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  or kp["width"]
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or kp["height"]
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or kp["T"]

    cmd = [
        "ffmpeg", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-",
        "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
        str(out_path),
    ]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                             stderr=subprocess.PIPE)
    try:
        t = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            kt = min(t + kp["frame_offset"], kp["T"] - 1)
            _draw_overlay(frame, kp, kt)
            proc.stdin.write(frame.tobytes())
            t += 1
            if progress is not None and n_frames:
                progress(min(1.0, t / n_frames))
    finally:
        cap.release()
        proc.stdin.close()
        stderr = proc.stderr.read()
        ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed: {stderr.decode(errors='replace')[-800:]}")


# ---------------------------------------------------------------------------
# Per-frame activation heatmaps — one collapsible pane per head, with a
# playhead line kept in sync with the shared <video> element client-side
# (no per-frame data round-trips to the server: it's purely a CSS left%
# derived from video.currentTime / video.duration).
# ---------------------------------------------------------------------------

def _pick_rows(probs: np.ndarray, n_show: int, pin_first: int | None) -> list[int]:
    C = probs.shape[1]
    max_act = probs.max(axis=0)
    others = [c for c in range(C) if c != pin_first]
    others.sort(key=lambda c: -max_act[c])
    rows = ([pin_first] if pin_first is not None else []) + others
    return rows[:n_show]


def _render_head_png(probs: np.ndarray, rows: list[int]) -> str:
    """Heatmap cells only — no axis, ticks or labels, and the Axes forced to
    fill the figure edge-to-edge (subplots_adjust to 0/1 margins). Row labels
    are rendered separately as an HTML sidebar (see _render_streams_html):
    baking variable-width label text into the image, as an earlier version
    did via ax.set_yticklabels + tight_layout, made the left margin (and so
    where x=0 actually falls in the image) depend on each head's longest
    label string — since different heads' label vocabularies have very
    different lengths, the playhead line (positioned as a plain left:X% of
    the image) was correct for some heads and visibly offset for others.
    An image that is always exactly the plot area, with no per-head-dependent
    margin, is what makes a single left:X% rule correct for every head.
    """
    fig, ax = plt.subplots(figsize=(10, max(1.0, len(rows) * 0.3)))
    ax.imshow(probs[:, rows].T, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1,
               interpolation="nearest")
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


# gr.HTML explicitly does not execute <script> tags embedded in its value
# (only static markup) — so none of this can be wired up per-render. Instead
# the CSS and all client-side behaviour live in the page <head> (via
# gr.Blocks(head=...), see HEAD_EXTRA below) and run once at page load:
#
#  - A persistent requestAnimationFrame loop drives the per-head playhead
#    lines AND a custom seek bar (native <input type="range"> — dragging it
#    is handled entirely by the browser, which tracks horizontal position
#    even once the pointer leaves the element, unlike Gradio's own video
#    scrubber). It re-queries the current video element, playhead divs and
#    seek bar fresh every frame rather than caching references, so it keeps
#    working across re-renders (new heatmaps after each Analyze run, a
#    possibly-recreated <video> node) with no reattachment logic needed.
#  - Left/Right arrow keys step the video by one frame (uses the clip's own
#    fps, embedded per-render in a hidden #stsnet-fps marker — see
#    _render_streams_html).
#  - A window-level dragover/drop preventDefault is a safety net against a
#    dropped file falling through to the browser's default "open as a new
#    page" behavior when it lands outside (or Gradio's own drop-zone isn't
#    live for) the video component.
HEAD_EXTRA = f"""
<style>
  .stsnet-head {{ border: 1px solid rgba(128,128,128,0.4); border-radius: 8px;
                  margin-bottom: 8px; overflow: hidden; }}
  .stsnet-head summary {{ cursor: pointer; padding: 8px 12px; font-weight: 600;
                           background: rgba(128,128,128,0.12); list-style: revert; }}
  .stsnet-panel {{ display: flex; align-items: stretch; }}
  .stsnet-labels {{ flex: 0 0 150px; display: flex; flex-direction: column; }}
  .stsnet-labels span {{ flex: 1; display: flex; align-items: center; justify-content: flex-end;
                          font-size: 11px; padding-right: 6px; white-space: nowrap;
                          overflow: hidden; text-overflow: ellipsis; }}
  .stsnet-imgwrap {{ position: relative; flex: 1 1 auto; line-height: 0; min-width: 0; }}
  .stsnet-imgwrap img {{ width: 100%; display: block; }}
  .stsnet-playhead {{ position: absolute; top: 0; bottom: 0; left: 0; width: 2px;
                       background: #e94560; pointer-events: none; }}
  #stsnet-seek {{ width: 100%; margin: 4px 0 10px; accent-color: #e94560; }}
</style>
<script>
(function() {{
  let scrubbing = false;

  function video() {{ return document.querySelector("#{VIDEO_ELEM_ID} video"); }}

  function ensureSeekBound() {{
    const seek = document.getElementById("stsnet-seek");
    if (seek && !seek.dataset.bound) {{
      seek.dataset.bound = "1";
      seek.addEventListener("pointerdown", function() {{ scrubbing = true; }});
      window.addEventListener("pointerup", function() {{ scrubbing = false; }});
      seek.addEventListener("input", function() {{
        const v = video();
        if (v && v.duration) v.currentTime = (seek.value / 1000) * v.duration;
      }});
    }}
    return seek;
  }}

  function tick() {{
    const v = video();
    const seek = ensureSeekBound();
    if (v && v.duration && !isNaN(v.duration)) {{
      const pct = Math.max(0, Math.min(1, v.currentTime / v.duration)) * 100;
      document.querySelectorAll(".stsnet-playhead").forEach(function(l) {{
        l.style.left = pct + "%";
      }});
      if (seek && !scrubbing) seek.value = pct * 10;
    }}
    requestAnimationFrame(tick);
  }}
  requestAnimationFrame(tick);

  document.addEventListener("keydown", function(e) {{
    if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
    const active = document.activeElement;
    if (active && (active.tagName === "INPUT" || active.tagName === "TEXTAREA")) return;
    const v = video();
    if (!v) return;
    const fpsEl = document.getElementById("stsnet-fps");
    const fps = fpsEl ? parseFloat(fpsEl.dataset.fps) : 25;
    const step = (isFinite(fps) && fps > 0) ? (1 / fps) : 0.04;
    const dur = v.duration || Infinity;
    v.currentTime = Math.max(0, Math.min(dur,
      v.currentTime + (e.key === "ArrowRight" ? step : -step)));
    e.preventDefault();
  }});

  ["dragover", "drop"].forEach(function(ev) {{
    window.addEventListener(ev, function(e) {{ e.preventDefault(); }}, false);
  }});
}})();
</script>
"""


def _render_streams_html(model_out: dict, fps: float) -> str:
    present = [h for h in HEAD_ORDER if h in model_out["heads"]]
    sections = [
        f'<input type="range" id="stsnet-seek" min="0" max="1000" value="0">'
        f'<div id="stsnet-fps" data-fps="{fps}" style="display:none"></div>'
    ]
    for h in present:
        probs  = np.array(model_out["heads"][h]["probs"])
        labels = model_out["heads"][h]["labels"]
        pin    = 0 if h in ("motion", "cloc", "ctype") else None
        rows   = _pick_rows(probs, N_SHOW.get(h, 10), pin)
        img_uri = _render_head_png(probs, rows)
        label_spans = "".join(
            f'<span title="{html.escape(labels[r])}">{html.escape(labels[r])}</span>'
            for r in rows
        )
        open_attr = " open" if h == "shape" else ""
        sections.append(
            f'<details{open_attr} class="stsnet-head">'
            f'<summary>{HEAD_TITLES.get(h, h)}</summary>'
            f'<div class="stsnet-panel">'
            f'<div class="stsnet-labels">{label_spans}</div>'
            f'<div class="stsnet-imgwrap"><img src="{img_uri}" draggable="false">'
            f'<div class="stsnet-playhead"></div></div>'
            f'</div></details>'
        )
    return "".join(sections)


# ---------------------------------------------------------------------------
# ffprobe helpers
# ---------------------------------------------------------------------------

def _probe_duration(video_path: Path) -> float | None:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(video_path)],
            capture_output=True, timeout=10,
        )
        return float(result.stdout.decode().strip())
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Gradio callbacks
# ---------------------------------------------------------------------------

@spaces.GPU(duration=30)
def _run_model(streams3d: dict):
    return _model_activations(MODEL_ENTRY, streams3d)


def run_demo(uploaded_path, show_overlay, prev_workdir, progress=gr.Progress()):
    if prev_workdir and os.path.isdir(prev_workdir):
        shutil.rmtree(prev_workdir, ignore_errors=True)

    if not uploaded_path:
        raise gr.Error("Please upload a video first.")
    video_path = Path(uploaded_path)

    progress(0.0, desc="Checking video...")
    duration = _probe_duration(video_path)
    if duration is None:
        raise gr.Error("Could not read this video file.")
    if duration > MAX_DURATION_S:
        raise gr.Error(
            f"This clip is {duration:.1f}s long; the demo is limited to "
            f"{MAX_DURATION_S:.0f}s so it stays responsive and within this "
            "free Space's daily GPU quota. Please trim it and try again."
        )

    workdir = Path(tempfile.mkdtemp(prefix="stsnet_demo_"))
    pose_path = workdir / "clip.pose"

    progress(0.05, desc="Extracting MediaPipe pose...")
    if not extract_pose(video_path, pose_path):
        raise gr.Error("Pose extraction failed for this video (no readable video stream?).")

    progress(0.5, desc="Loading pose streams...")
    streams3d = load_pose_streams(pose_path, HANDEDNESS, mirror_left=True)
    if streams3d is None:
        raise gr.Error("Could not load landmarks from this video — was a signer visible on camera?")

    progress(0.6, desc="Running STS-Net...")
    model_out = _run_model(streams3d)
    kp = _load_raw_keypoints(pose_path, video_path)

    progress(0.75, desc="Rendering activation heatmaps...")
    streams_html = _render_streams_html(model_out, kp["fps"])

    progress(0.8, desc="Rendering keypoint overlay...")
    overlay_path = workdir / "overlay.mp4"
    _build_overlay_video(video_path, kp, overlay_path,
                          progress=lambda f: progress(0.8 + 0.19 * f, desc="Rendering keypoint overlay..."))

    progress(1.0, desc="Done")
    display = str(overlay_path) if show_overlay else str(video_path)
    return display, streams_html, str(video_path), str(overlay_path), str(workdir)


def toggle_overlay(show_overlay, plain_path, overlay_path):
    if not plain_path:
        return gr.update()
    return overlay_path if show_overlay else plain_path


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

DESCRIPTION = """
# STS-Net — Swedish Sign Language phonology

Upload a short clip of someone signing and this model will predict eight
phonological properties (handshape, orientation, contact location/type,
motion direction, one-/two-handedness) frame by frame, and show how its
confidence evolves across the clip — expand a pane below to see a property's
per-frame activation heatmap, with a line tracking video playback.

Clips are capped at **{max_s:.0f} seconds** to keep this free Space
responsive. Pose is extracted with MediaPipe Holistic; nothing is stored
after your results are shown. See the
[GitHub repo](https://github.com/jbeskow/stsnet) for training code and model
details.
""".format(max_s=MAX_DURATION_S)

with gr.Blocks(title="STS-Net Demo", head=HEAD_EXTRA) as demo:
    gr.Markdown(DESCRIPTION)

    uploaded_state = gr.State()   # path of the most recently user-uploaded file
    plain_state    = gr.State()
    overlay_state  = gr.State()
    workdir_state  = gr.State()

    clip_name_out = gr.Markdown()
    video = gr.Video(elem_id=VIDEO_ELEM_ID, sources=["upload"],
                      label="Upload a sign-language clip, or drop one here")
    with gr.Row():
        overlay_toggle = gr.Checkbox(value=True, label="Show MediaPipe keypoint overlay")
        # Once `video` holds a result, dropping a replacement file onto it is
        # unreliable (Gradio's own drop-zone for Video doesn't consistently
        # stay live once the component has a value) — this button is a
        # guaranteed-to-work alternative for loading a different clip.
        new_clip_btn = gr.UploadButton("Load a different clip", file_types=["video"], size="sm")
    run_btn = gr.Button("Analyze", variant="primary")

    streams_out = gr.HTML()

    def _on_upload(path):
        name = f"### {Path(path).name}" if path else ""
        return path, name

    video.upload(fn=_on_upload, inputs=video, outputs=[uploaded_state, clip_name_out])
    new_clip_btn.upload(fn=lambda p: (p,) + _on_upload(p),
                         inputs=new_clip_btn, outputs=[video, uploaded_state, clip_name_out])

    run_btn.click(
        fn=run_demo,
        inputs=[uploaded_state, overlay_toggle, workdir_state],
        outputs=[video, streams_out, plain_state, overlay_state, workdir_state],
        concurrency_limit=1,
    )
    overlay_toggle.change(
        fn=toggle_overlay,
        inputs=[overlay_toggle, plain_state, overlay_state],
        outputs=[video],
    )

demo.queue(max_size=8)

if __name__ == "__main__":
    # ssr_mode defaults to an experimental on-by-default mode on recent
    # Gradio versions; disabled here after observing a first-load layout
    # oscillation that went away once the page's JS had fully hydrated
    # (e.g. after a manual window resize) — symptoms consistent with an
    # SSR/hydration mismatch. (Did not, on its own, fix drag-and-drop onto
    # an already-populated Video component — see new_clip_btn above.)
    demo.launch(ssr_mode=False)
