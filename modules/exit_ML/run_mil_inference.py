#!/usr/bin/env python3
"""
Inference helper for the attention-based MIL model.
Loads trained artefacts, applies the same preprocessing, and emits a JSON with
probability, top3 instances (timestamp/tx_hash + top7 feature values), and
bag-level FiLM top5 feature values.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import nn


DEFAULT_ARTIFACT_DIR = Path("./")
DEFAULT_FEATURE_CSV = Path("./features_exit_mil_new2.csv")
DEFAULT_STATIC_CSV = Path("./features_exit_static3.csv")
DEFAULT_TOKEN_INFO = Path("./token_information3.csv")

# Metadata columns to drop for model input but keep for reporting.
META_COLS = {
    "id",
    "instance_idx",
    "token_addr_idx",
    "event_time",
    "timestamp",
    "block_number",
    "tx_hash",
    "evt_idx",
}


class AttentionPooling(nn.Module):
    def __init__(self, input_dim: int, attention_dim: int):
        super().__init__()
        self.V = nn.Linear(input_dim, attention_dim)
        self.w = nn.Linear(attention_dim, 1, bias=False)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        scores = self.w(torch.tanh(self.V(h)))
        weights = torch.softmax(scores.squeeze(-1), dim=0).unsqueeze(-1)
        return weights


class FiLM(nn.Module):
    def __init__(self, instance_dim: int, static_dim: int):
        super().__init__()
        self.gamma = nn.Linear(static_dim, instance_dim)
        self.beta = nn.Linear(static_dim, instance_dim)
        nn.init.normal_(self.gamma.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.gamma.bias, 0.0)
        nn.init.normal_(self.beta.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.beta.bias, 0.0)

    def forward(self, instance_embed: torch.Tensor, static_features: torch.Tensor) -> torch.Tensor:
        scale = 1.0 + self.gamma(static_features)
        shift = self.beta(static_features)
        return instance_embed * scale + shift


class SequenceStaticAttentionMIL(nn.Module):
    def __init__(
        self,
        input_dim: int,
        static_dim: int,
        hidden_dim: int = 256,
        embed_dim: int = 128,
        attention_dim: int = 128,
        dropout: float = 0.2,
        film_scale: float = 1.0,
        instance_encoder: str = "mlp",
        bidirectional: bool = True,
    ):
        super().__init__()
        if not 0.0 <= film_scale <= 1.0:
            raise ValueError("film_scale must be in [0, 1].")
        self.encoder_type = instance_encoder
        if instance_encoder == "mlp":
            self.instance_encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, embed_dim),
                nn.ReLU(),
            )
            attn_input_dim = embed_dim
        elif instance_encoder == "gru":
            rnn_hidden = embed_dim
            self.instance_encoder = nn.GRU(
                input_dim,
                rnn_hidden,
                batch_first=True,
                bidirectional=bidirectional,
            )
            attn_input_dim = rnn_hidden * (2 if bidirectional else 1)
            self.instance_proj = nn.Sequential(nn.Linear(attn_input_dim, embed_dim), nn.ReLU())
            attn_input_dim = embed_dim
        else:
            raise ValueError("instance_encoder must be 'mlp' or 'gru'.")

        self.attention = AttentionPooling(attn_input_dim, attention_dim)
        self.film = FiLM(embed_dim, static_dim)
        self.film_scale = film_scale
        self.head_dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(embed_dim, 1)

    def forward(
        self,
        bags: Sequence[torch.Tensor],
        static_features: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        bag_embeddings: List[torch.Tensor] = []
        attention_weights: List[torch.Tensor] = []

        for bag in bags:
            if self.encoder_type == "mlp":
                h = self.instance_encoder(bag)
            else:
                seq = bag.unsqueeze(0)
                outputs, _ = self.instance_encoder(seq)
                h = outputs.squeeze(0)
                h = self.instance_proj(h)
            weights = self.attention(h)
            embedding = torch.sum(weights * h, dim=0)
            bag_embeddings.append(embedding)
            if return_attention:
                attention_weights.append(weights.squeeze(-1))

        bag_stack = torch.stack(bag_embeddings, dim=0)
        film_out = self.film(bag_stack, static_features)
        fused = self.film_scale * film_out + (1.0 - self.film_scale) * bag_stack
        fused = self.head_dropout(fused)
        logits = self.classifier(fused).squeeze(-1)
        if return_attention:
            return logits, attention_weights
        return logits, None


def _load_summary(artifact_dir: Path) -> Dict[str, object]:
    summary_path = artifact_dir / "attention_mil_training.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Training summary not found at {summary_path}")
    with summary_path.open() as fp:
        return json.load(fp)


def _load_importance(artifact_dir: Path, summary: Dict[str, object]) -> Dict[str, List[Dict[str, object]]]:
    artefacts = summary.get("artefacts") or {}
    path_hint = artefacts.get("feature_importance_path")
    if path_hint:
        path = Path(path_hint)
    else:
        path = artifact_dir / "attention_mil_feature_importance.json"
    if path.exists():
        with path.open() as fp:
            data = json.load(fp)
        return {
            "instance_top": data.get("instance_top") or [],
            "static_top": data.get("static_top") or [],
        }
    fallback = summary.get("feature_importance") or {}
    return {
        "instance_top": fallback.get("instance_top") or [],
        "static_top": fallback.get("static_top") or [],
    }


def _load_scaler(artifact_dir: Path) -> Dict[str, np.ndarray]:
    npz_path = artifact_dir / "attention_mil_scaler.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Scaler npz not found at {npz_path}")
    data = np.load(npz_path, allow_pickle=True)
    mean = data["mean"].astype(np.float32, copy=False)
    scale = data["scale"].astype(np.float32, copy=False)
    binary_cols = list(data["binary_cols"])
    return {"mean": mean, "scale": scale, "binary_cols": binary_cols}


def _ensure_feature_columns(frame: pd.DataFrame, feature_columns: List[str]) -> pd.DataFrame:
    for col in feature_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    extra = [c for c in frame.columns if c not in feature_columns]
    if extra:
        frame = frame.drop(columns=extra)
    return frame[feature_columns]


def _prepare_instance_features(
    df_raw: pd.DataFrame,
    feature_columns: List[str],
    scaler: Dict[str, np.ndarray],
) -> pd.DataFrame:
    features = df_raw.drop(columns=[c for c in META_COLS if c in df_raw.columns])
    features = features.replace({np.inf: np.nan, -np.inf: np.nan, "": np.nan})
    features = features.clip(lower=-1e12, upper=1e12)

    for col in list(features.columns):
        mask = features[col].isna()
        if mask.any():
            features[f"{col}_mask"] = (~mask).astype(float)
            features[col] = features[col].fillna(0.0)
    features = features.fillna(0.0)

    features = _ensure_feature_columns(features, feature_columns)
    values = features.to_numpy(dtype=np.float32, copy=False)
    mean = scaler["mean"]
    scale = scaler["scale"]
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    scaled = (values - mean) / safe_scale
    return pd.DataFrame(scaled, columns=feature_columns, index=df_raw.index)


def _prepare_static_features(
    static_path: Path,
    token_idx: int,
    static_columns: List[str],
    static_scaler: Dict[str, List[float]],
) -> np.ndarray:
    df_static = pd.read_csv(static_path)
    if "token_addr_idx" not in df_static.columns:
        raise ValueError("Static feature CSV must include token_addr_idx.")
    if token_idx not in set(df_static["token_addr_idx"].astype(int).tolist()):
        raise KeyError(f"token_addr_idx {token_idx} not found in static features.")

    frame = df_static.set_index("token_addr_idx")
    for col in static_columns:
        if col not in frame.columns:
            frame[col] = 0.0
    subset = frame[static_columns].apply(pd.to_numeric, errors="coerce")
    subset = subset.replace([np.inf, -np.inf], np.nan)
    subset = subset.mask(subset.abs() > 1e12)
    col_means = subset.mean(axis=0, skipna=True)
    subset = subset.fillna(col_means).fillna(0.0)

    row = subset.loc[token_idx].to_numpy(dtype=np.float32, copy=False)
    mean = np.asarray(static_scaler.get("mean", []), dtype=np.float32)
    scale = np.asarray(static_scaler.get("scale", []), dtype=np.float32)
    if mean.size != row.size or scale.size != row.size:
        raise ValueError("Static scaler shape mismatch with static feature columns.")
    safe_scale = np.where(scale == 0.0, 1.0, scale)
    return (row - mean) / safe_scale


def _select_top_instances(weights: np.ndarray, k: int = 3) -> List[int]:
    if weights.size == 0:
        return []
    k = min(k, weights.size)
    return np.argsort(weights)[::-1][:k].tolist()


def _make_instance_record(
    row: pd.Series,
    scaled_row: pd.Series,
    top_features: List[Dict[str, object]],
    rank: int,
) -> Dict[str, object]:
    ts = row.get("timestamp")
    if pd.isna(ts) and "event_time" in row:
        ts = row.get("event_time")
    if hasattr(ts, "isoformat"):
        ts = ts.isoformat()
    record: Dict[str, object] = {
        "timestamp": ts,
        "tx_hash": row.get("tx_hash"),
    }
    for idx, feat in enumerate(top_features, start=1):
        name = feat.get("feature")
        key = name if isinstance(name, str) and name else f"instance_feature_{idx}"
        val = scaled_row.get(name) if isinstance(name, str) and name in scaled_row else None
        record[key] = None if val is None else float(val)
    return record


def _make_static_record(
    static_top: List[Dict[str, object]],
    static_vector: np.ndarray,
    static_columns: List[str],
) -> Dict[str, object]:
    record: Dict[str, object] = {}
    for idx, feat in enumerate(static_top, start=1):
        name = feat.get("feature")
        key = name if isinstance(name, str) and name else f"bag_feature_{idx}"
        value = None
        if isinstance(name, str) and name in static_columns:
            pos = static_columns.index(name)
            if pos < static_vector.size:
                value = float(static_vector[pos])
        record[key] = value
    return record


def _token_addr_to_idx(token_addr: str, token_info_path: Path) -> int:
    df = pd.read_csv(token_info_path, usecols=["token_addr_idx", "token_addr"])
    df["token_addr"] = df["token_addr"].str.lower()
    token_addr = token_addr.lower()
    row = df.loc[df["token_addr"] == token_addr]
    if row.empty:
        raise KeyError(f"token_addr {token_addr} not found in {token_info_path}")
    return int(row.iloc[0]["token_addr_idx"])


def run_inference(
    token_addr: str,
    feature_path: Path,
    static_path: Path,
    token_info_path: Path,
    artifact_dir: Path,
    output_path: Optional[Path] = None,
) -> Dict[str, object]:
    summary = _load_summary(artifact_dir)
    feature_columns: List[str] = summary.get("feature_columns") or []
    static_columns: List[str] = summary.get("static_feature_columns") or []
    if not feature_columns:
        raise ValueError("feature_columns missing in training summary.")
    if not static_columns:
        raise ValueError("static_feature_columns missing in training summary.")

    scaler = _load_scaler(artifact_dir)
    static_scaler = summary.get("static_scaler") or {}
    importance = _load_importance(artifact_dir, summary)
    instance_top = importance.get("instance_top") or []
    static_top = importance.get("static_top") or []

    artefacts = summary.get("artefacts") or {}
    model_path = Path(artefacts.get("model_path") or (artifact_dir / "attention_mil_model.pt"))
    if not model_path.exists():
        fallback_model = artifact_dir / "attention_mil_model.pt"
        if fallback_model.exists():
            model_path = fallback_model
        else:
            raise FileNotFoundError(f"Model weights not found at {model_path}")
    params = summary.get("training_params") or {}

    token_idx = _token_addr_to_idx(token_addr, token_info_path)

    df_full = pd.read_csv(feature_path)
    df_full = df_full[df_full["token_addr_idx"] == token_idx]
    if df_full.empty:
        raise RuntimeError(f"No rows found for token_addr={token_addr} (idx={token_idx}) in {feature_path}")

    sort_cols = [c for c in ("token_addr_idx", "event_time", "evt_idx") if c in df_full.columns]
    if sort_cols:
        df_full = df_full.sort_values(sort_cols).reset_index(drop=True)

    scaled_instances = _prepare_instance_features(df_full, feature_columns, scaler)
    static_vec = _prepare_static_features(static_path, token_idx, static_columns, static_scaler)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SequenceStaticAttentionMIL(
        input_dim=len(feature_columns),
        static_dim=len(static_columns),
        hidden_dim=int(params.get("hidden_dim", 256)),
        embed_dim=int(params.get("embed_dim", 32)),
        attention_dim=int(params.get("attention_dim", 128)),
        dropout=float(params.get("dropout", 0.3)),
        film_scale=float(params.get("film_scale", 1.0)),
        instance_encoder=params.get("instance_encoder", "mlp"),
    ).to(device)
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bag_tensor = torch.from_numpy(scaled_instances.to_numpy(dtype=np.float32, copy=False)).to(device)
    static_tensor = torch.from_numpy(static_vec.astype(np.float32, copy=False)).unsqueeze(0).to(device)

    with torch.no_grad():
        logits, attn_list = model([bag_tensor], static_tensor, return_attention=True)
        prob = torch.sigmoid(logits.squeeze(0)).item()
        attn_weights = attn_list[0].detach().cpu().numpy() if attn_list else np.array([])

    top_positions = _select_top_instances(attn_weights, k=3)
    result: Dict[str, object] = {
        "token_addr": token_addr,
        "probability": float(prob),
    }

    for rank, pos in enumerate(top_positions, start=1):
        row = df_full.iloc[pos]
        scaled_row = scaled_instances.iloc[pos]
        record = _make_instance_record(row, scaled_row, instance_top, rank)
        result[f"detect_tx_{rank}"] = record

    if top_positions and len(top_positions) < 3:
        for rank in range(len(top_positions) + 1, 4):
            result[f"detect_tx_{rank}"] = None
    elif not top_positions:
        for rank in range(1, 4):
            result[f"detect_tx_{rank}"] = None

    static_record = _make_static_record(static_top, static_vec, static_columns)
    result["detect_static"] = static_record

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Run inference for a single token_addr.")
    parser.add_argument("--token-addr", type=str, required=True, help="Target token_addr (string).")
    parser.add_argument("--features-csv", type=Path, default=DEFAULT_FEATURE_CSV, help="Path to features_exit_mil CSV.")
    parser.add_argument("--static-csv", type=Path, default=DEFAULT_STATIC_CSV, help="Path to features_exit_static CSV.")
    parser.add_argument("--token-info", type=Path, default=DEFAULT_TOKEN_INFO, help="Path to token_information3.csv.")
    parser.add_argument("--artifact-dir", type=Path, default=DEFAULT_ARTIFACT_DIR, help="Directory with trained artefacts.")
    parser.add_argument("--output", type=Path, help="Optional path to write JSON output.")
    args = parser.parse_args(argv)

    result = run_inference(
        token_addr=args.token_addr,
        feature_path=args.features_csv,
        static_path=args.static_csv,
        token_info_path=args.token_info,
        artifact_dir=args.artifact_dir,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
