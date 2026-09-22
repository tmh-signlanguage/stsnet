"""
ClipClassifier: attention-pooled clip-level sign phonology model (STS-Net v0.2).

Architecture:
  FrameEncoder × N streams  (or DinoEncoder for flat feature streams)
  → Fusion MLP              (concat → 256-dim)
  → AttentionPool           (masked to sign-frame window)
  → Multi-label heads       (BCE for shape/att/cloc/ctype/motion; CE for hand_type)

The attention mask forces weights to zero outside the annotated sign range,
so the pooled embedding represents only the core signing portion of the clip.

Clip embeddings can be extracted via frame_features() for downstream tasks
(segmentation, retrieval) — returns per-frame fusion output (B, T, D).
"""

import math

import torch
import torch.nn as nn
from torch import Tensor

from stsnet.encoder import FrameEncoder, DinoEncoder


class AttentionPool(nn.Module):
    """
    Soft attention pooling with optional boolean mask.

    Scores = Linear(D→1) / sqrt(D).
    Frames outside the mask get -inf before softmax → weight ≈ 0.
    """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1, bias=False)
        self._scale = math.sqrt(hidden_dim)

    def forward(self, x: Tensor, mask: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """
        Args:
            x    : (B, T, D)
            mask : (B, T) bool — True = valid / in-sign frame

        Returns:
            clip_emb : (B, D)
            weights  : (B, T) attention weights
        """
        scores = self.score(x).squeeze(-1) / self._scale   # (B, T)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        weights = torch.softmax(scores, dim=-1)             # (B, T)
        weights = torch.nan_to_num(weights, nan=0.0)        # guard all-masked rows
        clip_emb = (weights.unsqueeze(-1) * x).sum(dim=1)  # (B, D)
        return clip_emb, weights


class ClipClassifier(nn.Module):
    """
    Clip-level multi-label phonology classifier (STS-Net v0.2).

    Args:
        num_shapes:       dominant handshape vocab size
        num_atts:         attitude vocab size
        num_cloc:         contact location vocab size
        num_ctype:        contact type vocab size
        num_motion:       motion direction vocab size (usually 8 incl. 'none')
        hidden_dim:       encoder channel width
        conv_layers:      temporal conv blocks per stream encoder
        kernel_size:      conv kernel size
        dropout:          dropout rate
        streams:          which input streams to use (dom always included)
        n_body:           body keypoint count (12 MediaPipe / 17 Sapiens)
        n_face:           face keypoint count (25 MediaPipe / 68 Sapiens)
        n_dims:           coordinate dims (3 = xyz, 2 = xy only)
        has_nondom_shape: add a non-dominant handshape prediction head
    """

    STREAM_JOINTS = {
        "dom": 21, "nondom": 21, "body": 12, "face": 25,
        "wilor_dom": 21, "wilor_nondom": 21,
        "dom_norm": 21, "nondom_norm": 21,
    }
    STREAM_FLAT = {"dino": 1024}   # flat (B, T, D) streams

    def __init__(
        self,
        num_shapes:       int,
        num_atts:         int,
        num_cloc:         int,
        num_ctype:        int,
        num_motion:       int = 8,
        hidden_dim:       int = 256,
        conv_layers:      int = 3,
        kernel_size:      int = 5,
        dropout:          float = 0.2,
        streams:          tuple[str, ...] = ("dom", "nondom", "body", "face"),
        n_body:           int = 12,
        n_face:           int = 25,
        n_dims:           int = 3,
        has_nondom_shape: bool = False,
        has_nondom_att:   bool = False,
    ):
        super().__init__()
        self.streams = tuple(dict.fromkeys(streams))
        self.n_dims = n_dims

        joint_override = {**self.STREAM_JOINTS, "body": n_body, "face": n_face}
        enc_kw = dict(hidden_dim=hidden_dim, conv_layers=conv_layers,
                      kernel_size=kernel_size, dropout=dropout, n_dims=n_dims)
        self.encoders = nn.ModuleDict()
        for s in self.streams:
            if s in joint_override:
                self.encoders[s] = FrameEncoder(n_joints=joint_override[s], **enc_kw)
            elif s in self.STREAM_FLAT:
                dino_kw = {k: v for k, v in enc_kw.items() if k != "n_dims"}
                self.encoders[s] = DinoEncoder(in_dim=self.STREAM_FLAT[s], **dino_kw)
            else:
                raise ValueError(f"Unknown stream '{s}'")

        D = hidden_dim
        self.fusion = nn.Sequential(
            nn.Linear(len(self.streams) * D, D),
            nn.LayerNorm(D),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.pool = AttentionPool(D)

        self.shape_head      = nn.Linear(D, num_shapes)
        self.att_head        = nn.Linear(D, num_atts)
        self.cloc_head       = nn.Linear(D, num_cloc)
        self.ctype_head      = nn.Linear(D, num_ctype)
        self.motion_head     = nn.Linear(D, num_motion)
        self.hand_type_head  = nn.Linear(D, 2)   # one / two (CE)

        self.has_nondom_shape = has_nondom_shape
        if has_nondom_shape:
            self.nondom_shape_head = nn.Linear(D, num_shapes)
        self.has_nondom_att = has_nondom_att
        if has_nondom_att:
            self.nondom_att_head = nn.Linear(D, num_atts)

    def _make_sign_mask(
        self,
        T:          int,
        sign_start: Tensor,   # (B,) int
        sign_end:   Tensor,   # (B,) int
        lengths:    Tensor,   # (B,) int
    ) -> Tensor:
        """Boolean mask (B, T): True for frames in [sign_start, sign_end) and not padding."""
        device = sign_start.device
        idx = torch.arange(T, device=device).unsqueeze(0)   # (1, T)
        in_sign = (idx >= sign_start.unsqueeze(1)) & (idx < sign_end.unsqueeze(1))
        not_pad = idx < lengths.unsqueeze(1)
        return in_sign & not_pad

    def frame_features(
        self,
        dominant:    Tensor,               # (B, T, 21, n_dims)
        nondominant: Tensor,               # (B, T, 21, n_dims)
        body:        Tensor,               # (B, T, n_body, n_dims)
        face:        Tensor | None = None, # (B, T, n_face, n_dims)
        **extra_streams: Tensor,
    ) -> Tensor:
        """Return per-frame fusion features (B, T, D) — useful for segmentation / retrieval."""
        n_face = (self.encoders["face"].frame_proj[0].in_features // self.n_dims
                  if "face" in self.encoders else 25)
        nan_face = torch.full((*dominant.shape[:2], n_face, self.n_dims), float("nan"),
                              device=dominant.device, dtype=dominant.dtype)
        stream_inputs = {
            "dom":    dominant,
            "nondom": nondominant,
            "body":   body,
            "face":   face if face is not None else nan_face,
            **extra_streams,
        }
        feats = [self.encoders[s](stream_inputs[s]) for s in self.streams]
        return self.fusion(torch.cat(feats, dim=-1))   # (B, T, D)

    def forward(
        self,
        dominant:    Tensor,               # (B, T, 21, n_dims)
        nondominant: Tensor,               # (B, T, 21, n_dims)
        body:        Tensor,               # (B, T, n_body, n_dims)
        face:        Tensor | None = None,
        sign_start:  Tensor | None = None, # (B,) int
        sign_end:    Tensor | None = None, # (B,) int
        lengths:     Tensor | None = None, # (B,) int
        **extra_streams: Tensor,
    ) -> dict[str, Tensor]:
        B, T = dominant.shape[:2]
        ctx = self.frame_features(dominant, nondominant, body, face, **extra_streams)

        if sign_start is not None and sign_end is not None:
            lens = lengths if lengths is not None else torch.full((B,), T, device=dominant.device)
            mask = self._make_sign_mask(T, sign_start, sign_end, lens)
        else:
            mask = None

        clip_emb, attn_weights = self.pool(ctx, mask)

        out = {
            "clip_emb":         clip_emb,
            "attn_weights":     attn_weights,
            "shape_logits":     self.shape_head(clip_emb),
            "att_logits":       self.att_head(clip_emb),
            "cloc_logits":      self.cloc_head(clip_emb),
            "ctype_logits":     self.ctype_head(clip_emb),
            "motion_logits":    self.motion_head(clip_emb),
            "hand_type_logits": self.hand_type_head(clip_emb),
        }
        if self.has_nondom_shape:
            out["nondom_shape_logits"] = self.nondom_shape_head(clip_emb)
        if self.has_nondom_att:
            out["nondom_att_logits"] = self.nondom_att_head(clip_emb)
        return out

    @classmethod
    def from_checkpoint(cls, ckpt_path: str, map_location: str = "cpu") -> tuple["ClipClassifier", dict]:
        """
        Load a ClipClassifier checkpoint saved by scripts/train_clip.py.

        Returns:
            model:      ClipClassifier in eval mode
            vocab_meta: dict with *_to_idx mappings
        """
        ck = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        kw = dict(ck["vocab_meta"]["model_kwargs"])
        kw["streams"] = tuple(kw["streams"])
        # Research checkpoints may carry kwargs this release does not implement
        # (e.g. tf_layers for the temporal-transformer variant); drop zero/False
        # ones silently, refuse anything that would change the architecture.
        import inspect
        known = set(inspect.signature(cls.__init__).parameters)
        extra = {k: v for k, v in kw.items() if k not in known}
        if any(v not in (0, False, None) for v in extra.values()):
            raise ValueError(f"checkpoint needs unsupported model options: {extra}")
        kw = {k: v for k, v in kw.items() if k in known}
        model = cls(**kw)
        model.load_state_dict(ck["model_state_dict"])
        model.eval()
        return model, ck["vocab_meta"]

    @classmethod
    def from_stsnet_checkpoint(cls, ckpt_path: str, **model_kwargs) -> "ClipClassifier":
        """
        Warm-start encoders + fusion from a v0.1 STSNet checkpoint.
        Pooling and classification heads are randomly initialised.
        """
        ck = torch.load(ckpt_path, map_location="cpu")
        model = cls(**model_kwargs)
        ck_sd = ck.get("model_state_dict", ck)
        own_sd = model.state_dict()

        loaded, skipped = 0, 0
        for key, val in ck_sd.items():
            if key in own_sd and own_sd[key].shape == val.shape:
                own_sd[key] = val
                loaded += 1
            else:
                skipped += 1

        model.load_state_dict(own_sd)
        print(f"Loaded {loaded} parameters from {ckpt_path} ({skipped} skipped)")
        return model
