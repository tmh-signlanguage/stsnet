# STS-Net

Phonological property prediction for Swedish Sign Language (STS) from pose keypoints.

**Current version: v0.2** — clip-level attention-pooled model (`ClipClassifier`).
Recommended checkpoint: **`checkpoints/stsnet_v02_mpp.pt`** (2D input, convolution +
3-layer transformer, masked-prediction pretraining, monotonic-alignment trained per-frame
phases — see [below](#transformer--masked-prediction-pretraining-stsnet_v02_mpppt)).
It is the most accurate checkpoint on both isolated dictionary signs (SSLL) and
continuous signing (SSLC). `checkpoints/stsnet_v02_mas.pt` is the smaller
convolution-only alternative (no transformer); it changes handshape more readily
from frame to frame, including more spurious changes.

v0.1 and v0.2 are incompatible model families and live in separate trees.
For the v0.1 per-frame BiLSTM model see [v0.1/README.md](v0.1/README.md).

**[Try it in your browser](https://huggingface.co/spaces/jbeskow/stsnet-demo)** —
upload a clip and see the per-frame streams below without installing
anything (Gradio Space on free ZeroGPU hardware; see [`demo/`](demo/)).

---

## What it predicts

Eight phonological properties, trained from **per-clip labels** and queryable
**per frame** at inference (see [Python API](#python-api)):

| Head | Target | Classes |
|------|--------|---------|
| shape (dom.) | multi-hot BCE | <details><summary>42 handshapes</summary>Flata handen · D-handen · Flata tumhanden · Sprethanden · 4-handen · Vinkelhanden · Tumvinkelhanden · A-handen · S-handen · Klohanden · O-handen · Knutna handen · E-handen · Tumhanden · Pekfingret · L-handen · Raka måtthanden · Nyphanden · Nyphanden (alt) · T-handen · Krokfingret · Måtthanden · Hållhanden · Långfingret · N-handen · Lilla O-handen · Lilla O-handen (alt) · V-handen · Tupphanden · K-handen · Dubbelkroken · Böjda tupphanden · M-handen · W-handen · Lillfingret · U-handen · Flyghanden · Stora långfingret · Runda långfingret · Stora nyphanden · Stora hållhanden · X-handen</details> |
| att (dom.) | multi-hot BCE | <details><summary>24 orientations</summary>framåtriktad-högervänd · framåtriktad-nedåtvänd · framåtriktad-uppåtvänd · framåtriktad-vänstervänd · högerriktad-framåtvänd · högerriktad-inåtvänd · högerriktad-nedåtvänd · högerriktad-uppåtvänd · inåtriktad-högervänd · inåtriktad-nedåtvänd · inåtriktad-uppåtvänd · inåtriktad-vänstervänd · nedåtriktad-framåtvänd · nedåtriktad-högervänd · nedåtriktad-inåtvänd · nedåtriktad-vänstervänd · uppåtriktad-framåtvänd · uppåtriktad-högervänd · uppåtriktad-inåtvänd · uppåtriktad-vänstervänd · vänsterriktad-framåtvänd · vänsterriktad-inåtvänd · vänsterriktad-nedåtvänd · vänsterriktad-uppåtvänd</details> |
| contact_loc | multi-hot BCE | <details><summary>22 locations</summary>none · mouth · chin · nose · forehead · cheek · ear · top_of_head · face · head · neck · chest · stomach · hip · shoulder · upper_arm · forearm · elbow · wrist · other_hand · temple · back</details> |
| contact_type | multi-hot BCE | <details><summary>4 types (9 manner classes in `stsnet_v02_mpp.pt`)</summary>none · single · repeated · sustained — `stsnet_v02_mpp.pt`: none · single · repeated · sustained · front · beside · above · below · near (the last five are non-contact locations relative to the body)</details> |
| motion | multi-hot BCE | <details><summary>8 directions</summary>none · nedåt · uppåt · framåt · bakåt · åt_höger · åt_vänster · inåt</details> |
| hand_type | CE | <details><summary>2 classes</summary>one · two</details> |
| nondom_shape | multi-hot BCE | <details><summary>42 handshapes (optional)</summary>Same vocabulary as shape (dom.)</details> |
| nondom_att | multi-hot BCE | <details><summary>24 orientations (optional)</summary>Same vocabulary as att (dom.)</details> |

Training uses one multi-hot target per clip — all properties that appear across
any phase of the sign are active. At inference the same heads can be applied to
individual frames, making it possible to track how predictions evolve through a
sign or to scan continuous video without pre-defined boundaries.

## Architecture

```mermaid
flowchart TD
    subgraph streams["Input  ·  T frames"]
        d["dominant hand\n21 kpts × 3D"]
        nd["non-dominant hand\n21 kpts × 3D"]
        b["upper body\n12 kpts × 3D"]
        f["face\n25 kpts × 3D"]
    end

    d  --> fe1["FrameEncoder\nLinear → 3× Conv1d residual\n→ T × 256"]
    nd --> fe2["FrameEncoder\nLinear → 3× Conv1d residual\n→ T × 256"]
    b  --> fe3["FrameEncoder\nLinear → 3× Conv1d residual\n→ T × 256"]
    f  --> fe4["FrameEncoder\nLinear → 3× Conv1d residual\n→ T × 256"]

    fe1 & fe2 & fe3 & fe4 --> fuse["Fusion MLP\nconcat T×1024  →  T×256"]

    fuse --> pool["Masked AttentionPool\nscores outside sign window → −∞\n→ 256-dim clip embedding"]

    subgraph dom["Dominant"]
        h1["shape\n42  multi-hot BCE"]
        h2["attitude\n24  multi-hot BCE"]
    end
    subgraph shared["Shared"]
        h3["contact location\n22  multi-hot BCE"]
        h4["contact type\n4  multi-hot BCE"]
        h5["motion\n8  multi-hot BCE"]
        h6["hand type\n2  CE"]
    end
    subgraph nondom["Non-dominant  (optional)"]
        h7["shape\n42  multi-hot BCE"]
        h8["attitude\n24  multi-hot BCE"]
    end

    pool --> dom
    pool --> shared
    pool --> nondom
```

Each stream passes through a `FrameEncoder` (linear projection → LayerNorm + ReLU → 3 × temporal Conv1d with residual connections). The four 256-dim per-frame outputs are concatenated and fused by a linear MLP, then aggregated by **masked attention pooling** — attention scores outside the annotated sign window are forced to −∞, so the pooled embedding represents only the core signing portion. Eight classification heads operate on the 256-dim clip embedding; the two non-dominant heads are optional and enabled at training time.

Optional additional streams: WiLoR 3D MANO joints (`wilor_dom`, `wilor_nondom`),
DINOv2 CLS tokens (`dino`), Moryossef rotation-normalised hands (`dom_norm`, `nondom_norm`),
or a Sapiens whole-body stream in place of MediaPipe.

Optionally (`tf_layers > 0`) a **temporal transformer** (pre-LN, 4 heads, sinusoidal
positions, padding frames masked) sits between fusion and pooling, giving every
frame global context over the clip; per-frame heads and the pool then read its
output. `stsnet_v02_mpp.pt` uses 3 layers.

Total parameters: ~4.3 M (6.66 M with 3 transformer layers).

## Installation

```bash
git clone git@github.com:jbeskow/stsnet.git
cd stsnet
git lfs pull          # download v0.1 checkpoint (~213 MB) if needed
pip install -e .
```

Requires Python 3.10+, PyTorch 2.0+, and [pose-format](https://github.com/sign-language-processing/pose-format).

---

## Python API

Although the model is **trained clip-level** (one prediction per sign), at
inference you can choose between clip-level and per-frame prediction modes.

```python
from stsnet import ClipClassifierInference

model = ClipClassifierInference("checkpoints/stsnet_v02_mpp.pt", device="cuda")
```

### Clip-level prediction (one label set per sign)

The attention pool aggregates frame features over the sign window into a single
256-dim embedding, and the heads predict one label per property:

```python
props = model.predict_phonology("clip.pose", sign_start=12, sign_end=58)
# {"shape": "Flata handen", "att": "vänsterriktad-nedåtvänd",
#  "cloc": "none", "ctype": "none", "motion": "none",
#  "hand_type": "one", "nondom_shape": "...", "nondom_att": "..."}
```

`sign_start` / `sign_end` are optional — if omitted, attention spans the whole clip.

### Per-frame prediction (one label set per frame)

The same classification heads are applied directly to the pre-pool per-frame
features, producing a label at every frame. Useful for visualising how
predictions evolve through a sign, or for scanning continuous video without
pre-defined sign boundaries:

```python
frames = model.predict_frames("continuous_video.pose")
# {"shape":  ["Flata handen", "Flata handen", ..., "Krokfingret", ...],
#  "att":    [...],
#  "motion": [...],
#  ...}   # one entry per frame, length = T
```

### Per-frame feature vectors

Raw 256-dim encoder features before pooling — suitable as input to a downstream
segmentation or sign-spotting head:

```python
feats = model.frame_features("continuous_video.pose")
# np.ndarray shape (T, 256)
```

### Clip embedding

256-dim pooled embedding for retrieval or clustering:

```python
emb = model.embed_clip("clip.pose", sign_start=12, sign_end=58)
# np.ndarray shape (256,)
```

---

## Command-line prediction

`scripts/predict.py` (`stsnet-predict`) runs inference on a `.pose` file or a
video (`.mp4`, `.mov`, `.avi`, `.mkv`, `.webm` — pose is extracted via
`video_to_pose` first) and writes CSV.

Per-frame predictions, CSV to stdout:

```bash
python scripts/predict.py clip.pose --ckpt checkpoints/stsnet_v02_mpp.pt
```

```
frame,shape,att,cloc,ctype,motion,hand_type,nondom_shape,nondom_att
0,Flata handen,vänsterriktad-nedåtvänd,none,none,none,one,...
1,Flata handen,vänsterriktad-nedåtvänd,none,none,none,one,...
```

Clip-level prediction for a single sign (`--sign_start`/`--sign_end`), written to a file:

```bash
python scripts/predict.py clip.pose --sign_start 12 --sign_end 58 --out preds.csv
```

Per-frame encoder embeddings `(T, 256)` alongside the CSV, saved as a `.pt` tensor:

```bash
python scripts/predict.py clip.pose --out preds.csv --embeddings feats.pt
```

Other flags: `--heads` (restrict to specific heads), `--start`/`--end` (frame
range for per-frame output), `--handedness`, `--device`.

---

## Interactive inspector

For a no-install version with a single video, see the
[hosted demo](https://huggingface.co/spaces/jbeskow/stsnet-demo)
(`demo/`) — it shows the clip-level prediction, the MediaPipe keypoint
overlay, and the same per-frame heatmap/attention/entropy streams as the
tool below, for one upload at a time. For browsing a local batch of clips or
comparing checkpoints side by side, run the desktop inspector:

`scripts/inspector.py` (`stsnet-inspect`) is a small Flask web GUI for
browsing predictions on a handful of clips: a file list, a video player, and
a collapsible per-frame activation heatmap for every head (plus the
attention trace), synced to the video playhead. Takes video or `.pose` files
directly on the command line — no CSV or dataset metadata required.

```bash
python scripts/inspector.py clip1.mp4 clip2.mp4 clip3.pose \
    --ckpt checkpoints/stsnet_v02.pt --port 7860
```

Open `http://localhost:7860/`, pick a clip from the sidebar, and click a
section header to collapse/expand its heatmap. Videos are pose-extracted
once (via `video_to_pose`) and cached alongside the source file, or under
`--pose_cache_dir` if given; `.pose` files are used as-is.

More clips can be added at any time by dragging video (or `.pose`) files
onto the browser window — each is uploaded with a live progress bar, then
pose-extracted in the background (shown as an "Extracting pose…" row) and
added to the sidebar once ready. The command-line `videos` argument is
optional, so the server can also be started empty (`python scripts/inspector.py
--ckpt ...`) and populated entirely by drag-and-drop. Uploaded files are
saved under `--upload_dir` (default: a temp directory); pass `--no_upload`
to disable the feature.

**Comparing models.** Pass several checkpoints to `--ckpt` (with optional
`--names`) to inspect two or more models on the same clips:

```bash
python scripts/inspector.py clip1.mp4 clip2.mp4 \
    --ckpt checkpoints/stsnet_v02_noz.pt checkpoints/stsnet_v02_mas.pt \
    --names AP MAS
```

Each head section then stacks one heatmap per model on a shared time axis,
with the same class rows (chosen by label across all models, so differing
vocabularies still line up) and, above each heatmap, that model's per-frame
argmax strip — its decoded phase sequence. Per-frame probabilities are
sigmoid for attention-pooled (BCE) checkpoints and softmax for
alignment-trained (`objective: mas_*`) ones, read from the checkpoint's
`vocab_meta`. 2D/3D input is likewise detected per checkpoint.

A "Show keypoints" toggle above the video overlays the raw MediaPipe Holistic
landmarks (body pose, both hands, and a reduced set of face points) directly
on the video frame, color-coded per stream and synced to the playhead —
useful for sanity-checking that pose extraction tracked the signer
correctly. The overlay uses each clip's un-normalized pixel-space landmarks
(a separate `/api/clip/<idx>/keypoints` endpoint), independent of the
model-input streams used for the activation heatmaps.

An "Entropy" section alongside the attention trace plots per-frame predictive
entropy (bits/class, mean over classes for each multi-hot head so every head
sits on the same 0–1 scale, then averaged across all active heads) — low
values mean the per-frame heads are confidently committing to a (sub)set of
classes, high values mean the prediction is closer to a coin flip. This is a
diagnostic on the per-frame heads applied directly to fusion features, which
bypass the trained attention pool (see `ClipInference.predict_frames`), not a
quantity the model was directly supervised on. It's exposed as a candidate
signal for keyframe/segmentation exploration; a per-head breakdown is
available in the same API response (`entropy_by_head`) for clips where a
single property's transitions are of interest.

---

## Training

### Prerequisites

- SSLL dataset: `sign_data_with_signer_fr.csv`, `.pose` files, `pseudo_signing.json`
  (see [v0.1/README.md](v0.1/README.md) for pose extraction and pseudo-signing steps)
- Pose cache built with `cache_poses.py` (see CLAUDE.md)

### Train on SSLL only

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/train_clip.py \
    --out runs/clip_v02
```

Key options:

| Flag | Default | Description |
|------|---------|-------------|
| `--streams` | `dom nondom body face` | Input streams to use |
| `--hidden_dim` | 256 | Encoder channel width |
| `--epochs` | 60 | Training epochs |
| `--dropout` | 0.2 | Dropout rate |
| `--no_z` | off | Use 2D (xy) pose only |
| `--nondom_shape_head` | off | Enable non-dominant handshape head |
| `--nondom_att_head` | off | Enable non-dominant attitude head |
| `--ckpt` | None | Warm-start encoders from v0.1 checkpoint |
| `--wilor_dir` | None | Add WiLoR 3D hand streams |
| `--dino_dir` | None | Add DINOv2 CLS token stream |

Checkpoint saved to `runs/clip_v02/best.pt` (includes vocab metadata).

### Mine SSLC and re-train

Transfer phonological labels from SSLL to unannotated continuous-signing data
via nearest-neighbour embedding matching:

```bash
# Step 1: mine SSLC
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/mine_sslc.py \
    --clip_ckpt runs/clip_v02/best.pt \
    --threshold 0.4 \
    --out runs/sslc_mined.json

# Step 2: re-train with mined data + stronger regularisation
CUDA_VISIBLE_DEVICES=0 conda run -n slp python -u scripts/train_clip.py \
    --out runs/clip_v02_mined \
    --clip_ckpt runs/clip_v02/best.pt \
    --mined_json runs/sslc_mined.json \
    --dropout 0.3 --noise_std 0.02 --label_smoothing 0.1 \
    --time_stretch_min 0.85 --time_stretch_max 1.15
```

Mining selects SSLC gloss windows whose embedding (cosine distance) to the nearest
SSLL training variant is below `--threshold`. Mined instances carry the matched
SSLL phonological targets.

### Performance (MediaPipe baseline, SSLL val set)

Clip-level accuracy (top-1 prediction ∈ the clip's label set), all clips:

| Property | SSLL only | +mined SSLC (3D) | +mined SSLC (2D) | 2D + monotonic alignment |
|----------|-----------|-----------------|-----------------|------------------------|
| Handshape (dom.) | 77.8% | 84.7% | 85.4% | **88.0%** |
| Attitude (dom.) | 75.2% | 78.8% | **80.2%** | 78.9% |
| Contact location | — | — | **80.4%** | 79.1% |
| Contact type | — | — | **75.8%** | 70.9% |
| Motion direction | — | — | **64.2%** | 58.9% |
| Hand type | 96.6% | 96.9% | **97.3%** | 95.0% |
| Handshape (nondom.) | 80.7% | 86.2% | 86.0% | **87.5%** |
| Attitude (nondom.) | 75.6% | 78.8% | 79.2% | **79.4%** |

SSLL-only = `clip_nd_att_base` (ep 29). 3D = `stsnet_v02.pt` / `clip_nd_att_combined_v3` (17,833 SSLC clips, best-val checkpoint).
2D = `stsnet_v02_noz.pt` / `clip_nd_att_combined_v3_noz_reg2` (same data, 2D xy only, ep 60 last checkpoint).
2D + monotonic alignment = `stsnet_v02_mas.pt` / `clip_mas_pool` (see below).

#### Per-frame phase sequences: `stsnet_v02_mas.pt`

The attention-pooled models are trained on the *set* of properties in a clip and
ignore their order, so their per-frame heads (`predict_frames`) flicker and carry
no phase structure. `stsnet_v02_mas.pt` has the same architecture, data and
input (2D) but was trained with a **monotonic alignment** between the sign-window
frames and the *ordered* phases of the dictionary description (Viterbi /
Monotonic Alignment Search, then per-frame cross-entropy on the aligned phase
labels), jointly with the usual pooled objective. It is a drop-in replacement
for `stsnet_v02_noz.pt` in every script: `predict_phonology` / `embed_clip` use
its (trained) attention pool, `predict_frames` gives temporally coherent,
ordered phase predictions.

Handshape phase-sequence metrics on SSLL val (per-frame argmax in the sign
window, run-length decoded, vs. the ordered phases from the description;
n = 4043 clips, 1320 of them multi-phase):

| | `stsnet_v02_noz.pt` (AP) | `stsnet_v02_mas.pt` |
|---|---|---|
| exact sequence match, all clips | 2.7% | **46.5%** |
| exact sequence match, multi-phase clips | 0.9% | **29.1%** |
| normalised edit distance ↓ | 0.646 | **0.317** |
| last phase correct, multi-phase clips | 4.8% | **72.8%** |
| phase-count error (MAE, phases) | 2.43 | **0.95** |

Same picture for attitude, contact location and motion. Among the conv-only
checkpoints, `stsnet_v02_mas.pt` is the one to use whenever the per-frame output matters; `stsnet_v02_noz.pt` remains ~1–5 pts
better on clip-level attitude / contact / motion / hand type.

#### Transformer + masked-prediction pretraining: `stsnet_v02_mpp.pt`

Monotonic-alignment model like `stsnet_v02_mas.pt`, with three changes:

1. **Temporal transformer** (3 layers) after fusion — 6.66 M parameters.
2. **Masked phonological prediction pretraining** (HuBERT-style). On ~49 h of
   unlabelled continuous SSLC, a frozen STS-Net teacher labels every frame
   (softmax of all phase heads); the student sees the same window with 45 % of
   frames masked in 5–15-frame spans (whole frame, or only the dominant hand) and
   learns to predict the teacher's distributions at the masked frames from
   context. It is then fine-tuned with the usual labelled recipe (SSLL + mined
   SSLC, alignment + pooled objective, 40 epochs). This is iterated once,
   HuBERT-style: the fine-tuned model becomes the teacher for a second
   pretraining run, on SSLC plus ~20 h of Swedish YouTube-SL-25 video (69 h of
   unlabelled pose in total), followed by the same fine-tune.
3. **Location + manner labels**: corrected contact-location parsing, and the
   contact-type head becomes a 9-way *manner* head (none / single / repeated /
   sustained contact, plus non-contact front / beside / above / below / near).

SSLL val (same split as above), clip-level top-1-in-set:

| Property | `stsnet_v02_mas.pt` | `stsnet_v02_mpp.pt` |
|---|---|---|
| Handshape (dom.) | 88.0% | **89.4%** |
| Attitude (dom.) | 78.9% | **81.7%** |
| Contact location (v4 targets) | 77.7%¹ | **83.6%** |
| Manner (9 classes, v4 targets) | 67.8%¹ | **75.5%** |
| Motion direction (v4 targets) | 58.9% | **70.4%** |
| Hand type | 95.0% | **96.6%** |
| Handshape (nondom.) | 87.5% | **88.1%** |
| Attitude (nondom.) | 79.4% | **81.7%** |

¹ For contact location and manner the comparison is against the same
conv-only alignment model retrained on the v3 labels (scored on v3 targets).

**v4 labels (2026-10-29).** The checkpoint is fine-tuned on
label version v4: a grammar-based coverage check of the SSLL descriptions found
parts the v3 parser dropped, now parsed — shape changes with adverbials
("förändras först/sedan till X", chained changes), bare motion directions
("förs upp/fram/in"), "kontakt med <handform>" as the non-dominant hand, more
contact keywords and body locations, a second contact location ("… och sedan
på X"), per-hand attitudes ("den högra … den vänstra …") and "riktade/vända från
varandra". Targets change in ~5.7k SSLL clips; class inventories are unchanged,
so the interface is identical to the previous `stsnet_v02_mpp.pt`. Against the
previous checkpoint on v4 targets: contact location +2.9, manner +2.9, motion
+2.4, nondom. attitude +1.6, nondom. handshape +0.9 pts; handshape and attitude
unchanged. On clips with multi-step shape changes it gets the last shape right
59 % of the time (previous checkpoint 41 %).

Manner recall improves most on the classes the earlier model rarely got right:
repeated 36 → 63 %, non-contact locations 4–13 → 22–48 %. Over replicate
fine-tunes the second pretraining iteration adds ≈ +1 pt on location, manner
and motion (run-to-run spread ≈ 0.5–1 pt) and is neutral on hand shape and
attitude.

Handshape phase sequences: exact match 70.4 % of all clips (vs 46.5 %), 40.4 % of
multi-phase clips (vs 29.1 %), normalised edit distance 0.180 (vs 0.317),
last phase of multi-phase clips 80.4 % (vs 72.8 %); phase-count MAE 0.27 for the v4 checkpoint (vs 0.95).

**Continuous signing (current checkpoint, 2026-10-30).** The previous
transformer checkpoints under-segmented running signing: the mined SSLC clips
were cut exactly to the gloss span, so the transformer never saw neighbouring
signs in training and let them dominate at inference. The current checkpoint is
fine-tuned with up to ±4 s (random 0–100 frames per side) of the surrounding
continuous video around each mined clip; the losses still apply only to the
gloss span. The SSLL numbers above are for this checkpoint; against the
previous one they are within ±1 pt.

Evaluation on SSLC gloss instances that were never in training (16,110
single-shape / 3,898 multi-shape glosses whose SSLL entry has one unambiguous
handshape sequence; the model runs continuously over the whole video):

| | `stsnet_v02_mas.pt` | previous `mpp` | `stsnet_v02_mpp.pt` |
|---|---|---|---|
| right shape, single-shape glosses | 46.5 % | 32.3 % | **52.5 %** |
| right shape, multi-shape glosses | 42.3 % | 28.5 % | **54.6 %** |
| first shape right, multi-shape glosses | 12.6 % | 8.2 % | **25.9 %** |
| spurious changes in single-shape glosses | 32 % | 13 % | 18 % |
| same, held-out signers only (single / multi) | 43.0 / 43.8 % | 29.5 / 29.7 % | **50.3 / 58.7 %** |

On 15 held-out-signer SSLC videos, its per-frame handshape changes track
hand-pose change about as well as the conv model (Spearman ρ 0.42 vs 0.44, equal
precision, 1.5 vs 1.9 changes/s; the previous checkpoint had ρ 0.27, 0.8
changes/s).

The training code for this checkpoint (pretraining, v3/v4 labels) currently lives
in the research repository; `scripts/train_clip.py` here trains the conv-only
variant.

---

## Repository layout

This tree holds v0.2 only. The incompatible v0.1 model lives standalone
under [`v0.1/`](v0.1/) — its own `stsnet` package, scripts, checkpoint,
data, and `pip install -e .`; see [v0.1/README.md](v0.1/README.md).

```
stsnet/                      v0.2 package
  __init__.py                exports ClipClassifier, ClipClassifierInference
  clip_classifier.py         ClipClassifier + AttentionPool + TemporalTransformer  ← default model
  encoder.py                 FrameEncoder, DinoEncoder, TemporalConvBlock
  inference.py               ClipClassifierInference
  data/
    pose_io.py               MediaPipe loading, load_wilor_streams, normalize_hand_moryossef
    ssll_clip.py              SSLLClipDataset, collate_clip, load_sapiens_streams
    sslc_mined.py            SSLCMinedDataset (mined SSLC gloss clips)
    description.py           Swedish sign description parser
    contact.py               contact location / type vocabularies
scripts/
  train_clip.py              train ClipClassifier       (stsnet-train-clip)
  mine_sslc.py               mine SSLC via embedding matching (stsnet-mine)
  predict.py                 CSV predictions + per-frame embeddings (stsnet-predict)
  inspector.py               interactive activation-heatmap web GUI (stsnet-inspect)
checkpoints/
  stsnet_v02.pt               pretrained v0.2 checkpoint — 3D (xyz) input (Git LFS)
  stsnet_v02_noz.pt           pretrained v0.2 checkpoint — 2D (xy only) input (Git LFS)
  stsnet_v02_mas.pt           v0.2, 2D, monotonic-alignment trained: ordered per-frame phases — conv-only (Git LFS)
  stsnet_v02_mpp.pt           v0.2, 2D, + transformer, masked-prediction pretraining, manner labels, continuous context — recommended (Git LFS)
data/
  sts_handformer.txt         handshape vocabulary (42 classes)
v0.1/                        standalone v0.1 tree — own package, scripts, checkpoint, data
```

---

## v0.1 model

The v0.1 per-frame BiLSTM model (`STSNet`) predicts nine phonological features
per frame and supports Viterbi forced alignment for timeline annotation. It is
a separate, incompatible package rooted at [`v0.1/`](v0.1/) — see
[v0.1/README.md](v0.1/README.md) for full documentation and the pretrained
checkpoint at `v0.1/checkpoints/stsnet_base.pt`.

---

## Publication

Jonas Beskow, *STS-net — a phonological property prediction model for Swedish
sign language*. Nordic Prosody XIV, Stockholm, 18–20 August 2026.
[PDF](docs/stsnet_nordic_prosody_26.pdf)
