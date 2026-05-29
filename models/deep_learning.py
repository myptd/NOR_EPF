"""
Deep learning models for day-ahead electricity price forecasting.

Three architectures:
  lstm        — multi-layer LSTM
  tcn         — Temporal Convolutional Network with dilated causal convolutions
  transformer — Transformer encoder with sinusoidal positional encoding

All models:
  - Input  : last SEQ_LEN=168 hours of all features (batch × seq × features)
  - Output : next PRED_HORIZON=24 hours of price (multi-step direct), but only step 0 (next hour) is evaluated to keep apples-to-apples with LightGBM/Ridge ARX
  - Same train/val/test split (train<2024, val=2024, test=2025)

Normalization:
  Global StandardScaler (fit on train only) is applied before training.  Models
  use standard per-layer norms: GroupNorm for TCN, LayerNorm for Transformer,
  LayerNorm input for LSTM.  No instance normalization is needed because the data
  is already well-scaled by StandardScaler.

Loss: L1 (MAE) — consistent with LightGBM/XGBoost objective and the primary
  evaluation metric, keeping the benchmarking comparison apples-to-apples.

Linear shortcut: each model adds a direct Linear(input_size → horizon) path
  from the last observed feature vector.  This encodes the linear-ARX baseline
  as an inductive bias and accelerates convergence — the sequence encoder only
  needs to learn the nonlinear residual.

Optimizer: AdamW + linear warmup (5 epochs) + cosine annealing.
Early stopping: patience=20 on validation L1 loss.
stride=6 training windows (≈7k sequences vs ≈1.8k with stride=24).

TCN receptive field: 1 + 2*(kernel_size-1)*sum([1,2,4,8,16,32,64])
  = 1 + 2*2*127 = 509 > SEQ_LEN=168.

Model weights saved to model_weights/{model}_{zone}.pt.
Supports MPS (Apple Silicon), CUDA, and CPU.

Run:
    python models/deep_learning.py --model tcn         [--zones NO1 ...]
    python models/deep_learning.py --model transformer [--zones NO1 ...]
    python models/deep_learning.py --model lstm        [--zones NO1 ...]
"""
from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.utils import (
    load_zone, build_feature_matrix,
    train_val_test_split, evaluate, print_results,
    SEED, set_seed, save_results,
)

TARGET       = "price_eur_mwh"
SEQ_LEN      = 168   # 1 week of hourly context
PRED_HORIZON = 24    # multi-step training auxiliary (24 output heads);
                     # only step 0 (next-hour) is used for evaluation — same task as LightGBM/Ridge ARX

if torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class PriceSequenceDataset(Dataset):
    """
    Sliding-window dataset.

    X[i] = feature_matrix[s : s+seq_len]
    y[i] = target[s+seq_len : s+seq_len+horizon]
    """
    def __init__(self, X: np.ndarray, y: np.ndarray,
                 seq_len: int = SEQ_LEN, horizon: int = PRED_HORIZON,
                 stride: int = 24):
        self.X      = torch.from_numpy(X.astype(np.float32))
        self.y      = torch.from_numpy(y.astype(np.float32))
        self.seq    = seq_len
        self.h      = horizon
        self.stride = stride
        max_start   = len(X) - seq_len - horizon
        self.starts = list(range(0, max_start + 1, stride))

    def __len__(self) -> int:
        return len(self.starts)

    def __getitem__(self, i: int):
        s = self.starts[i]
        return self.X[s : s + self.seq], self.y[s + self.seq : s + self.seq + self.h]


# ---------------------------------------------------------------------------
# Shared forecasting head (used by all three architectures)
# ---------------------------------------------------------------------------

def _make_head(in_size: int, horizon: int, dropout: float) -> nn.Sequential:
    """LayerNorm -> Linear -> GELU -> Dropout -> Linear."""
    mid = max(in_size // 2, horizon * 2)
    return nn.Sequential(
        nn.LayerNorm(in_size),
        nn.Linear(in_size, mid),
        nn.GELU(),
        nn.Dropout(dropout),
        nn.Linear(mid, horizon),
    )


# ---------------------------------------------------------------------------
# Model 1 - LSTM
# ---------------------------------------------------------------------------

class LSTMForecaster(nn.Module):
    """
    Multi-layer LSTM with a direct linear shortcut.

    output = shortcut(X[:, -1, :]) + lstm_head(X)

    The shortcut encodes the linear-ARX baseline as an inductive bias;
    the LSTM learns only the nonlinear residual. A LayerNorm on the
    input provides stability without instance-level normalisation.
    Input X is already GlobalStandardScaler-normalised.
    """
    def __init__(self, input_size: int, hidden_size: int = 128,
                 num_layers: int = 6, dropout: float = 0.15,
                 horizon: int = PRED_HORIZON):
        super().__init__()
        self.input_norm = nn.LayerNorm(input_size)
        self.shortcut   = nn.Linear(input_size, horizon)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = _make_head(hidden_size, horizon, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n    = self.input_norm(x)             # (B, seq, F)
        skip   = self.shortcut(x_n[:, -1, :])  # linear shortcut (B, H)
        out, _ = self.lstm(x_n)
        return skip + self.head(out[:, -1, :])  # shortcut + residual


# ---------------------------------------------------------------------------
# Model 2 - Temporal Convolutional Network (TCN)
# ---------------------------------------------------------------------------

class _CausalConv1d(nn.Module):
    """Causal 1-D convolution: left-only padding, no look-ahead."""
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, dilation: int = 1):
        super().__init__()
        self._pad = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size,
                              padding=self._pad, dilation=dilation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv(x)
        return y[:, :, :-self._pad] if self._pad else y


class _TCNResBlock(nn.Module):
    """
    Two causal convolutions + residual skip (Bai et al. 2018, WaveNet style).
    GroupNorm (g=8) is more robust than BatchNorm under distribution shift
    and works for small batch sizes.
    """
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int,
                 dilation: int, dropout: float):
        super().__init__()
        g = min(8, out_ch)
        self.net = nn.Sequential(
            _CausalConv1d(in_ch,  out_ch, kernel_size, dilation),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
            _CausalConv1d(out_ch, out_ch, kernel_size, dilation),
            nn.GroupNorm(g, out_ch),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.skip = nn.Conv1d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x) + self.skip(x)


class TCNForecaster(nn.Module):
    """
    TCN with exponentially growing dilations + direct linear shortcut.

    Receptive field (kernel_size=3, 6 dilation levels [1,2,4,8,16,32]):
        1 + 2*(3-1)*sum([1,2,4,8,16,32]) = 253 > SEQ_LEN=168.

    Input X is GlobalStandardScaler-normalised; output is in the same
    standardised target space. Caller applies scaler_y.inverse_transform()
    to recover EUR/MWh.
    """
    DILATIONS = [1, 2, 4, 8, 16, 32]

    def __init__(self, input_size: int, d_model: int = 128,
                 kernel_size: int = 3, dropout: float = 0.1,
                 horizon: int = PRED_HORIZON):
        super().__init__()
        self.shortcut = nn.Linear(input_size, horizon)
        self.proj     = nn.Conv1d(input_size, d_model, 1)
        blocks: list[nn.Module] = []
        for dil in self.DILATIONS:
            blocks.append(_TCNResBlock(d_model, d_model, kernel_size, dil, dropout))
        self.blocks = nn.Sequential(*blocks)
        self.head   = _make_head(d_model, horizon, dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skip = self.shortcut(x[:, -1, :])      # linear shortcut (B, H)
        y    = self.proj(x.transpose(1, 2))    # (B, d_model, seq)
        y    = self.blocks(y)
        return skip + self.head(y[:, :, -1])   # shortcut + residual


# ---------------------------------------------------------------------------
# Model 3 - Transformer encoder
# ---------------------------------------------------------------------------

class _SinusoidalPE(nn.Module):
    """Fixed sinusoidal positional encoding (Vaswani et al. 2017)."""
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        pe  = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float()
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.drop(x + self.pe[:, :x.size(1)])


class TransformerForecaster(nn.Module):
    """
    Transformer encoder (pre-LayerNorm) + direct linear shortcut.

    A learnable [CLS] token prepended to the sequence aggregates temporal
    context for the forecasting head (Devlin et al. 2019).
    Input X is GlobalStandardScaler-normalised; output is in the same
    standardised target space.
    """
    def __init__(self, input_size: int, d_model: int = 128, nhead: int = 4,
                 num_layers: int = 6, dim_ff: int = 256, dropout: float = 0.1,
                 horizon: int = PRED_HORIZON):
        super().__init__()
        assert d_model % nhead == 0, "d_model must be divisible by nhead"
        self.shortcut = nn.Linear(input_size, horizon)
        self.proj     = nn.Linear(input_size, d_model)
        self.pos      = _SinusoidalPE(d_model, max_len=SEQ_LEN + 1, dropout=dropout)
        self.cls      = nn.Parameter(torch.zeros(1, 1, d_model))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=dim_ff,
            dropout=dropout, batch_first=True,
            norm_first=True,    # pre-norm for training stability
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers,
                                             enable_nested_tensor=False)
        self.head = _make_head(d_model, horizon, dropout)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b    = x.size(0)
        skip = self.shortcut(x[:, -1, :])              # linear shortcut (B, H)
        e    = self.proj(x)                            # (B, seq, d_model)
        cls  = self.cls.expand(b, -1, -1)              # (B, 1, d_model)
        e    = self.pos(torch.cat([cls, e], dim=1))    # (B, 1+seq, d_model) + PE
        e    = self.encoder(e)
        return skip + self.head(e[:, 0])               # CLS token -> head


# ---------------------------------------------------------------------------
# Training - shared across all architectures
# ---------------------------------------------------------------------------

def _warmup_cosine_schedule(optimiser, warmup_steps: int, total_steps: int):
    """Linear warmup then cosine annealing to eta_min=1e-6."""
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(1e-6, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return torch.optim.lr_scheduler.LambdaLR(optimiser, lr_lambda)


def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader:   DataLoader,
    epochs:        int   = 150,
    lr:            float = 5e-4,
    patience:      int   = 20,
    warmup_epochs: int   = 5,
) -> nn.Module:
    """
    AdamW + linear warmup + cosine annealing.
    L1 loss (MAE) -- consistent with LightGBM/XGBoost objective and primary metric.
    Early stopping on validation MAE.
    """
    optimiser    = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    total_steps  = epochs * len(train_loader)
    warmup_steps = warmup_epochs * len(train_loader)
    scheduler    = _warmup_cosine_schedule(optimiser, warmup_steps, total_steps)
    criterion    = nn.L1Loss()   # MAE loss -- matches primary evaluation metric

    best_val   = float("inf")
    best_state: dict | None = None
    no_improve = 0

    model.to(DEVICE)
    for epoch in range(1, epochs + 1):
        # ---- train ----
        model.train()
        tr_loss = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimiser.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
            scheduler.step()
            tr_loss += loss.item() * len(xb)
        tr_loss /= len(train_loader.dataset)

        # ---- validate ----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                val_loss += criterion(model(xb), yb).item() * len(xb)
        val_loss /= len(val_loader.dataset)

        if epoch % 10 == 0 or epoch == 1:
            print(f"    Epoch {epoch:3d}/{epochs}  "
                  f"train={tr_loss:.4f}  val={val_loss:.4f}  "
                  f"lr={optimiser.param_groups[0]['lr']:.2e}", flush=True)

        if val_loss < best_val - 1e-5:
            best_val   = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"    Early stopping at epoch {epoch}.", flush=True)
                break

    if best_state:
        model.load_state_dict(best_state)
    return model


# ---------------------------------------------------------------------------
# Prediction - rebuild hourly series from sliding windows
# ---------------------------------------------------------------------------

def predict_sequence(
    model:    nn.Module,
    dataset:  PriceSequenceDataset,
    scaler_y: StandardScaler,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (y_true, y_pred) in original EUR/MWh scale — next-hour (step 0) only.

    The model outputs PRED_HORIZON=24 steps per window.  Using stride=1 on
    val/test and selecting only step 0 ([:, 0]) gives one non-overlapping
    next-hour prediction per hour, making DL evaluation directly comparable
    to LightGBM and Ridge ARX (both one-step-ahead).

    The 24-step training objective acts as an auxiliary task that encourages
    the model to learn multi-horizon structure, but the final metric is the
    same next-hour MAE as all other models.
    """
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for xb, yb in loader:
            preds.append(model(xb.to(DEVICE)).cpu().numpy())
            trues.append(yb.numpy())

    preds = np.concatenate(preds)   # (N, horizon)
    trues = np.concatenate(trues)   # (N, horizon)

    # Inverse-transform from standardised space -> EUR/MWh
    preds_eur = scaler_y.inverse_transform(preds.reshape(-1, 1)).reshape(preds.shape)
    trues_eur = scaler_y.inverse_transform(trues.reshape(-1, 1)).reshape(trues.shape)

    # Step 0 = next-hour prediction; all models are evaluated on this single step
    return trues_eur[:, 0], preds_eur[:, 0]


# ---------------------------------------------------------------------------
# Model factory & per-model defaults
# ---------------------------------------------------------------------------

_MODEL_DEFAULTS = {
    # d_model/hidden=128: ~400-700k params -- enough for nonlinear patterns,
    # small enough to avoid overfitting on ~43k training hours.
    "lstm": dict(
        epochs=150, lr=5e-4, patience=20, train_stride=6,
        hidden=128, num_layers=6, dropout=0.15,
    ),
    "tcn": dict(
        epochs=150, lr=5e-4, patience=20, train_stride=6,
        hidden=128, num_layers=6, dropout=0.1,
    ),
    "transformer": dict(
        epochs=150, lr=3e-4, patience=20, train_stride=6,
        hidden=128, num_layers=6, dropout=0.1,
    ),
}


def build_model(model_name: str, input_size: int, hidden: int,
                num_layers: int, dropout: float) -> nn.Module:
    if model_name == "lstm":
        return LSTMForecaster(input_size, hidden_size=hidden,
                              num_layers=num_layers, dropout=dropout)
    elif model_name == "tcn":
        return TCNForecaster(input_size, d_model=hidden,
                             kernel_size=3, dropout=dropout)
    elif model_name == "transformer":
        nhead  = 4 if hidden <= 128 else 8
        dim_ff = hidden * 2  # compact FF (256 for d_model=128)
        return TransformerForecaster(
            input_size, d_model=hidden, nhead=nhead,
            num_layers=num_layers, dim_ff=dim_ff, dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name!r}. Choose lstm | tcn | transformer")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(
    data_path:    str   = "data/cleaned/NO1_hourly.parquet",
    model_name:   str   = "tcn",
    val_start:    str   = "2024-01-01",
    test_start:   str   = "2025-01-01",
    epochs:       int   = 150,
    seq_len:      int   = SEQ_LEN,
    batch_size:   int   = 256,
    hidden:       int   = 128,
    num_layers:   int   = 2,
    dropout:      float = 0.1,
    lr:           float = 5e-4,
    train_stride: int   = 6,
) -> dict:
    set_seed()
    zone  = Path(data_path).stem.replace("_hourly", "")
    label = model_name.upper()
    print(f"\n[{label}] Loading {data_path} ...", flush=True)
    df = load_zone(data_path)

    print(f"[{label}] Building feature matrix ...", flush=True)
    X_df, y_s = build_feature_matrix(df, target=TARGET)
    X_tr_df, X_val_df, X_te_df, y_tr_s, y_val_s, y_te_s = train_val_test_split(
        X_df, y_s, val_start=val_start, test_start=test_start)
    print(f"  Train: {len(y_tr_s):,}  Val: {len(y_val_s):,}  Test: {len(y_te_s):,}  "
          f"Features: {X_df.shape[1]}  Device: {DEVICE}", flush=True)

    # Fit scalers on train only -- strict temporal split, no leakage
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_tr  = scaler_X.fit_transform(X_tr_df.values)
    X_val = scaler_X.transform(X_val_df.values)
    X_te  = scaler_X.transform(X_te_df.values)
    y_tr  = scaler_y.fit_transform(y_tr_s.values.reshape(-1, 1)).ravel()
    y_val = scaler_y.transform(y_val_s.values.reshape(-1, 1)).ravel()
    y_te  = scaler_y.transform(y_te_s.values.reshape(-1, 1)).ravel()

    torch.manual_seed(SEED)
    model    = build_model(model_name, X_tr.shape[1], hidden, num_layers, dropout)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  {label} parameters: {n_params:,}", flush=True)

    # stride=6 -> ~7x more training sequences than stride=24
    ds_tr  = PriceSequenceDataset(X_tr,  y_tr,  seq_len=seq_len, stride=train_stride)
    ds_val = PriceSequenceDataset(X_val, y_val, seq_len=seq_len, stride=1)
    ds_te  = PriceSequenceDataset(X_te,  y_te,  seq_len=seq_len, stride=1)
    print(f"  Train sequences: {len(ds_tr):,} (stride={train_stride})", flush=True)

    tr_loader  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  num_workers=0)
    val_loader = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=0)

    patience = _MODEL_DEFAULTS[model_name]["patience"]
    print(f"\n[{label}] Training ({epochs} epochs max, patience={patience}) ...",
          flush=True)
    model = train_model(model, tr_loader, val_loader,
                        epochs=epochs, lr=lr, patience=patience)

    # Evaluate on val and test sets
    print(f"\n[{label}] Evaluating ...", flush=True)
    y_val_true, y_val_pred = predict_sequence(model, ds_val, scaler_y)
    y_te_true,  y_te_pred  = predict_sequence(model, ds_te,  scaler_y)

    val_res  = evaluate(y_val_true, y_val_pred, label)
    test_res = evaluate(y_te_true,  y_te_pred,  label)

    print(f"\n--- {label} Validation (2024) ---")
    print_results([val_res])
    print(f"--- {label} Test (2025) ---")
    print_results([test_res])

    # Persist weights and CSV results
    out = Path(f"model_weights/{model_name}_{zone}.pt")
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out)
    print(f"  Weights saved -> {out}")
    save_results([test_res], f"{zone}_{model_name}_test")
    save_results([val_res],  f"{zone}_{model_name}_val")

    return {"val": val_res, "test": test_res}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train deep learning models for electricity price forecasting.")
    parser.add_argument("--model",        default="tcn",
                        choices=["lstm", "tcn", "transformer"],
                        help="Architecture to train (default: tcn)")
    parser.add_argument("--data",         default="data/cleaned/NO1_hourly.parquet")
    parser.add_argument("--zones",        nargs="+", default=None,
                        help="Zones to run, e.g. --zones NO1 NO2 NO3 NO4 NO5")
    parser.add_argument("--val-start",    default="2024-01-01")
    parser.add_argument("--test-start",   default="2025-01-01")
    parser.add_argument("--epochs",       type=int,   default=None)
    parser.add_argument("--seq-len",      type=int,   default=SEQ_LEN)
    parser.add_argument("--batch-size",   type=int,   default=256)
    parser.add_argument("--hidden",       type=int,   default=None,
                        help="d_model / hidden size (model-specific default if omitted)")
    parser.add_argument("--num-layers",   type=int,   default=None)
    parser.add_argument("--dropout",      type=float, default=None)
    parser.add_argument("--lr",           type=float, default=None)
    parser.add_argument("--train-stride", type=int,   default=6)
    args = parser.parse_args()

    defs         = _MODEL_DEFAULTS[args.model]
    epochs       = args.epochs     or defs["epochs"]
    lr           = args.lr         or defs["lr"]
    hidden       = args.hidden     or defs["hidden"]
    num_layers   = args.num_layers or defs["num_layers"]
    dropout      = args.dropout    if args.dropout is not None else defs["dropout"]
    train_stride = args.train_stride

    zones = args.zones or [Path(args.data).stem.replace("_hourly", "")]
    for z in zones:
        main(
            data_path    = f"data/cleaned/{z}_hourly.parquet",
            model_name   = args.model,
            val_start    = args.val_start,
            test_start   = args.test_start,
            epochs       = epochs,
            seq_len      = args.seq_len,
            batch_size   = args.batch_size,
            hidden       = hidden,
            num_layers   = num_layers,
            dropout      = dropout,
            lr           = lr,
            train_stride = train_stride,
        )
