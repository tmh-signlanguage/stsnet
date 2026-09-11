"""
Inference API for STS-Net v0.2.

ClipClassifierInference: clip-level phonological prediction and embedding.
"""

from pathlib import Path

import numpy as np
import torch

from stsnet.data.pose_io import load_pose_streams


class ClipClassifierInference:
    """
    High-level inference API for a trained ClipClassifier checkpoint.

    Args:
        checkpoint_path: Path to a best.pt checkpoint saved by scripts/train_clip.py.
        device:          Torch device string (e.g. "cpu", "cuda:0").
        no_z:            Strip the z-coordinate from pose streams before inference.
                         ``None`` (default) auto-detects from the checkpoint:
                         2D models (n_dims=2) strip z automatically, 3D models
                         keep all three coordinates.  Pass ``True`` to force 2D
                         even on a 3D-trained model, or ``False`` to keep 3D
                         regardless of the checkpoint.
    """

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "cpu",
        no_z: bool | None = None,
    ):
        from stsnet.clip_classifier import ClipClassifier
        self.device = torch.device(device)
        self.model, self.vocab_meta = ClipClassifier.from_checkpoint(
            str(checkpoint_path), map_location=device
        )
        self.model.to(self.device)
        self.model.eval()

        # Determine whether to strip z: explicit override > checkpoint n_dims
        ckpt_n_dims = self.vocab_meta.get("model_kwargs", {}).get("n_dims", 3)
        if no_z is None:
            self.no_z = (ckpt_n_dims == 2)
        else:
            self.no_z = bool(no_z)
        self.n_dims = 2 if self.no_z else 3

        self._idx_to_shape  = {v: k for k, v in self.vocab_meta["shape_to_idx"].items()}
        self._idx_to_att    = {v: k for k, v in self.vocab_meta["att_to_idx"].items()}
        self._idx_to_motion = {v: k for k, v in self.vocab_meta.get("motion_to_idx", {}).items()}
        self._idx_to_cloc   = {v: k for k, v in self.vocab_meta.get("cloc_to_idx", {}).items()}
        self._idx_to_ctype  = {v: k for k, v in self.vocab_meta.get("ctype_to_idx", {}).items()}

    def _load_streams(
        self,
        pose_path:   str | Path,
        handedness:  str = "right",
        f0:          int = 0,
        f1:          int | None = None,
    ) -> dict[str, torch.Tensor] | None:
        """Load pose streams, optionally sliced to [f0, f1), as (1, T, J, C) tensors.

        C is 2 when no_z is True (or the checkpoint was trained with n_dims=2),
        otherwise C is 3.
        """
        streams = load_pose_streams(Path(pose_path), handedness, mirror_left=True)
        if streams is None:
            return None
        if f1 is not None:
            streams = {k: v[f0:f1] for k, v in streams.items()}
        elif f0 > 0:
            streams = {k: v[f0:] for k, v in streams.items()}
        if self.no_z:
            streams = {k: v[..., :2] for k, v in streams.items()}
        return {
            k: torch.from_numpy(v).unsqueeze(0).to(self.device)
            for k, v in streams.items()
        }

    def predict_phonology(
        self,
        pose_path:   str | Path,
        sign_start:  int | None = None,
        sign_end:    int | None = None,
        handedness:  str = "right",
    ) -> dict[str, str] | None:
        """
        Predict phonological properties for a single sign.

        If sign_start/sign_end are provided the model's attention is masked to that
        window; otherwise attention spans the whole clip.

        Returns:
            dict with keys: shape, att, cloc, ctype, motion, hand_type,
                            nondom_shape (if available)
            Returns None if pose loading fails.
        """
        streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None

        T = streams["dominant"].shape[1]
        kwargs = {}
        if sign_start is not None and sign_end is not None:
            kwargs["sign_start"] = torch.tensor([sign_start], device=self.device)
            kwargs["sign_end"]   = torch.tensor([sign_end],   device=self.device)
            kwargs["lengths"]    = torch.tensor([T],          device=self.device)

        with torch.no_grad():
            out = self.model(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
                **kwargs,
            )

        rev_maps = {
            "shape_logits":        self._idx_to_shape,
            "att_logits":          self._idx_to_att,
            "motion_logits":       self._idx_to_motion,
            "cloc_logits":         self._idx_to_cloc,
            "ctype_logits":        self._idx_to_ctype,
            "hand_type_logits":    {0: "one", 1: "two"},
            "nondom_shape_logits": self._idx_to_shape,
            "nondom_att_logits":   self._idx_to_att,
        }
        result = {}
        key_names = {
            "shape_logits":      "shape",
            "att_logits":        "att",
            "cloc_logits":       "cloc",
            "ctype_logits":      "ctype",
            "motion_logits":     "motion",
            "hand_type_logits":  "hand_type",
            "nondom_shape_logits": "nondom_shape",
            "nondom_att_logits":   "nondom_att",
        }
        for logit_key, name in key_names.items():
            if logit_key not in out:
                continue
            idx = int(out[logit_key].squeeze(0).argmax())
            rev = rev_maps.get(logit_key, {})
            result[name] = rev.get(idx, str(idx))
        return result

    def embed_clip(
        self,
        pose_path:   str | Path,
        sign_start:  int | None = None,
        sign_end:    int | None = None,
        handedness:  str = "right",
    ) -> np.ndarray | None:
        """
        Return the 256-dim clip embedding for a sign, useful for retrieval or clustering.

        If sign_start/sign_end are given, slice the pose to that window (the
        attention mask is then set to cover all frames of the slice).
        Returns None if pose loading fails.
        """
        if sign_start is not None and sign_end is not None:
            streams = self._load_streams(pose_path, handedness, sign_start, sign_end)
        else:
            streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None

        T = streams["dominant"].shape[1]
        with torch.no_grad():
            out = self.model(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
                sign_start=torch.zeros(1, dtype=torch.long, device=self.device),
                sign_end=torch.tensor([T], dtype=torch.long, device=self.device),
                lengths=torch.tensor([T], dtype=torch.long, device=self.device),
            )
        return out["clip_emb"].squeeze(0).cpu().numpy()

    def predict_frames(
        self,
        pose_path:  str | Path,
        handedness: str = "right",
    ) -> dict[str, list[str]] | None:
        """
        Per-frame phonological predictions over an entire clip.

        The classification heads are applied directly to the per-frame fusion
        features (before attention pooling), producing a label at every frame.
        This is useful for visualising how predictions evolve through a sign,
        or for scanning continuous video without pre-defined sign boundaries.

        Returns:
            dict mapping head name → list of T label strings.
            Returns None if pose loading fails.
        """
        streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None

        with torch.no_grad():
            frame_feats = self.model.frame_features(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
            )   # (1, T, D)
            f = frame_feats.squeeze(0)   # (T, D)
            logits = {
                "shape":     self.model.shape_head(f),
                "att":       self.model.att_head(f),
                "cloc":      self.model.cloc_head(f),
                "ctype":     self.model.ctype_head(f),
                "motion":    self.model.motion_head(f),
                "hand_type": self.model.hand_type_head(f),
            }
            if self.model.has_nondom_shape:
                logits["nondom_shape"] = self.model.nondom_shape_head(f)
            if self.model.has_nondom_att:
                logits["nondom_att"] = self.model.nondom_att_head(f)

        rev_maps = {
            "shape":     self._idx_to_shape,
            "att":       self._idx_to_att,
            "motion":    self._idx_to_motion,
            "cloc":      self._idx_to_cloc,
            "ctype":     self._idx_to_ctype,
            "hand_type": {0: "one", 1: "two"},
            "nondom_shape": self._idx_to_shape,
            "nondom_att":   self._idx_to_att,
        }
        result = {}
        for name, lgt in logits.items():
            indices = lgt.argmax(dim=-1).cpu().tolist()
            rev = rev_maps.get(name, {})
            result[name] = [rev.get(i, str(i)) for i in indices]
        return result

    def frame_features(
        self,
        pose_path:  str | Path,
        handedness: str = "right",
    ) -> np.ndarray | None:
        """
        Raw per-frame encoder features (T, 256) before attention pooling.

        Useful as input to a downstream segmentation or spotting head.
        Returns None if pose loading fails.
        """
        streams = self._load_streams(pose_path, handedness)
        if streams is None:
            return None
        with torch.no_grad():
            feats = self.model.frame_features(
                streams["dominant"], streams["nondominant"],
                streams["body"], streams.get("face"),
            )
        return feats.squeeze(0).cpu().numpy()   # (T, 256)
