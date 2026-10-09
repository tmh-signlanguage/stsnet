---
title: STS-Net Demo
emoji: 🤟
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 5.49.1
app_file: app.py
pinned: false
license: mit
python_version: "3.12"
---

# STS-Net — Swedish Sign Language phonology

Upload a short sign-language clip and this Space will extract MediaPipe
Holistic pose, run it through **STS-Net v0.2** (`ClipClassifier`), and show
the model's clip-level phonological predictions alongside the same
per-frame diagnostic streams as the project's desktop inspector: per-head
activation heatmaps, the trained attention trace, and a predictive-entropy
trace.

Runs on a free ZeroGPU Space (shared GPU attached only for the model's
forward pass; pose extraction runs on CPU). Clips are capped to stay within
the daily free GPU quota; nothing uploaded is stored beyond the current
session.

Code, training scripts, and the full desktop inspector (`stsnet-inspect`,
which supports comparing multiple checkpoints side by side) live at
[github.com/jbeskow/stsnet](https://github.com/jbeskow/stsnet). This Space
is built from that repo's `demo/` directory.

## Layout uploaded to this Space

```
app.py              demo/app.py
requirements.txt    demo/requirements.txt
packages.txt         demo/packages.txt
stsnet/              stsnet/            (model + data pipeline package)
scripts/              scripts/           (incl. inspector.py, reused for inference helpers)
pyproject.toml        pyproject.toml     (lets `pip install .` register the stsnet package)
checkpoints/stsnet_v02.pt   checkpoints/stsnet_v02.pt
```
