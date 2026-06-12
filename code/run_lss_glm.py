#!/usr/bin/env python
"""用 Least-Squares-Separate (LSS) GLM 提取 trial-level beta。

核心思路：
1. 每个 trial 单独作为目标回归量；
2. 同一个 run 中其他 trial 合并为一个“其他刺激”回归量；
3. 同时加入 fMRIPrep confounds、DCT 高通漂移和截距；
4. 输出 4D beta 图像，第四维顺序与 trial 元数据表完全一致。
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
import re
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from scipy.stats import gamma


EVENT_RE = re.compile(
    r"(?P<sub>sub-[A-Za-z0-9]+)_task-(?P<task>[A-Za-z0-9]+)_run-(?P<run>[0-9]+)_events\.tsv$"
)


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def spm_hrf(tr: float, oversampling: int = 16, time_length: float = 32.0) -> np.ndarray:
    """生成近似 SPM canonical HRF。"""
    dt = tr / oversampling
    t = np.arange(0, time_length, dt)
    hrf = gamma.pdf(t, 6) - gamma.pdf(t, 16) / 6
    hrf = hrf / np.sum(hrf)
    return hrf.astype(np.float64)


def events_to_regressor(
    onsets: np.ndarray,
    durations: np.ndarray,
    n_scans: int,
    tr: float,
    oversampling: int = 16,
) -> np.ndarray:
    """把事件 onset/duration 卷积 HRF 后采样到 TR 时间点。"""
    dt = tr / oversampling
    n_hr = int(math.ceil((n_scans * tr + 32.0) / dt))
    stim = np.zeros(n_hr, dtype=np.float64)
    for onset, duration in zip(onsets, durations):
        start = max(int(round(float(onset) / dt)), 0)
        stop = max(int(round((float(onset) + float(duration)) / dt)), start + 1)
        stop = min(stop, n_hr)
        stim[start:stop] += 1.0
    conv = np.convolve(stim, spm_hrf(tr, oversampling=oversampling))[:n_hr]
    sample_idx = np.round(np.arange(n_scans) * tr / dt).astype(int)
    sample_idx = np.clip(sample_idx, 0, len(conv) - 1)
    return conv[sample_idx]


def make_dct_drift(n_scans: int, tr: float, high_pass_hz: float) -> np.ndarray:
    """生成 DCT 高通漂移基函数，近似 nilearn/spm 的做法。"""
    if high_pass_hz <= 0:
        return np.empty((n_scans, 0), dtype=np.float64)
    frame_times = np.arange(n_scans) * tr
    period_cut = 1.0 / high_pass_hz
    order = int(np.floor(2 * (frame_times[-1] - frame_times[0]) / period_cut))
    if order <= 0:
        return np.empty((n_scans, 0), dtype=np.float64)
    drift = np.zeros((n_scans, order), dtype=np.float64)
    n = float(n_scans)
    for k in range(1, order + 1):
        drift[:, k - 1] = np.cos((np.pi / n) * (np.arange(n_scans) + 0.5) * k)
    return drift


def standardize_columns(x: np.ndarray) -> np.ndarray:
    """标准化 nuisance 回归量；常数列会被丢弃。"""
    if x.size == 0:
        return x
    x = np.asarray(x, dtype=np.float64)
    x = x - np.nanmean(x, axis=0, keepdims=True)
    sd = np.nanstd(x, axis=0, keepdims=True)
    keep = np.squeeze(sd > 1e-8)
    if keep.ndim == 0:
        keep = np.array([bool(keep)])
    x = x[:, keep]
    sd = sd[:, keep]
    return x / sd


def select_confounds(confounds_path: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """按配置选择 fMRIPrep confounds，缺失列自动跳过。"""
    df = pd.read_csv(confounds_path, sep="\t")
    selected: list[str] = []
    for col in cfg.get("default_confounds", []):
        if col in df.columns:
            selected.append(col)

    # CompCor 列很多，默认每个前缀只取前 N 个，避免模型过拟合。
    n_compcor = int(cfg.get("n_compcor_per_prefix", 6))
    for prefix in cfg.get("compcor_prefixes", []):
        cols = sorted([c for c in df.columns if c.startswith(prefix)])[:n_compcor]
        selected.extend(cols)

    selected = list(dict.fromkeys(selected))
    out = df[selected].copy() if selected else pd.DataFrame(index=df.index)
    out = out.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return out


def find_preproc_bold(func_dir: Path, sub: str, task: str, run: str, space: str) -> Path | None:
    exact = sorted(
        func_dir.glob(f"{sub}_task-{task}_run-{run}_space-{space}*_desc-preproc_bold.nii.gz")
    )
    if exact:
        return exact[0]
    fallback = sorted(func_dir.glob(f"{sub}_task-{task}_run-{run}_*_desc-preproc_bold.nii.gz"))
    return fallback[0] if fallback else None


def matching_mask(preproc_bold: Path) -> Path:
    candidate = Path(str(preproc_bold).replace("_desc-preproc_bold.nii.gz", "_desc-brain_mask.nii.gz"))
    if not candidate.exists():
        raise FileNotFoundError(f"找不到与 beta 输入对应的 brain mask: {candidate}")
    return candidate


def gzip_integrity_ok(path: Path) -> bool:
    """确认 .nii.gz 压缩流完整，避免截断文件进入 GLM。"""
    if not str(path).endswith(".gz"):
        return True
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (EOFError, OSError, gzip.BadGzipFile):
        return False


def iter_runs(bids_dir: Path, fmriprep_dir: Path, task: str, space: str, subject: str | None) -> list[dict[str, Path | str]]:
    """找出当前已经具备 LSS 输入文件的 run。"""
    pattern = f"{subject or 'sub-*'}/func/*_task-{task}_run-*_events.tsv"
    runs = []
    for events_path in sorted(bids_dir.glob(pattern)):
        match = EVENT_RE.match(events_path.name)
        if not match:
            continue
        sub = match.group("sub")
        run = match.group("run")
        func_dir = fmriprep_dir / sub / "func"
        preproc = find_preproc_bold(func_dir, sub, task, run, space)
        if preproc is None:
            continue
        mask = matching_mask(preproc)
        if not gzip_integrity_ok(preproc) or not gzip_integrity_ok(mask):
            print(f"跳过压缩流不完整的 run: {sub} run-{run}")
            continue
        confounds = func_dir / f"{sub}_task-{task}_run-{run}_desc-confounds_timeseries.tsv"
        if not confounds.exists():
            continue
        runs.append(
            {
                "subject": sub,
                "run": run,
                "events": events_path,
                "preproc": preproc,
                "mask": mask,
                "confounds": confounds,
            }
        )
    return runs


def fit_lss_for_run(run_info: dict[str, Path | str], cfg: dict[str, Any], out_dir: Path, overwrite: bool) -> tuple[Path, Path]:
    sub = str(run_info["subject"])
    run = str(run_info["run"])
    events_path = Path(run_info["events"])
    preproc_path = Path(run_info["preproc"])
    mask_path = Path(run_info["mask"])
    confounds_path = Path(run_info["confounds"])
    task = str(cfg.get("task", "tactile"))
    space = str(cfg.get("space", "T1w"))

    sub_out = out_dir / sub / "func"
    sub_out.mkdir(parents=True, exist_ok=True)
    beta_path = sub_out / f"{sub}_task-{task}_run-{run}_space-{space}_desc-lssTrialBetas_beta.nii.gz"
    meta_path = sub_out / f"{sub}_task-{task}_run-{run}_space-{space}_desc-lssTrialBetas_events.tsv"
    design_path = sub_out / f"{sub}_task-{task}_run-{run}_space-{space}_desc-lssDesignSummary.json"
    if beta_path.exists() and meta_path.exists() and not overwrite:
        print(f"跳过已存在 LSS beta: {beta_path}")
        return beta_path, meta_path

    img = nib.load(str(preproc_path))
    mask_img = nib.load(str(mask_path))
    if img.shape[:3] != mask_img.shape[:3]:
        raise ValueError(f"BOLD 与 mask 维度不一致: {preproc_path} vs {mask_path}")

    tr = float(cfg.get("tr") or img.header.get_zooms()[3])
    n_scans = int(img.shape[3])
    trim_start = int(cfg.get("trim_start_tr", 0))
    trim_end = int(cfg.get("trim_end_tr", 0))
    keep_slice = slice(trim_start, n_scans - trim_end if trim_end else n_scans)
    kept_n = len(range(*keep_slice.indices(n_scans)))
    if kept_n <= 10:
        raise ValueError(f"保留 TR 数过少: {kept_n}")

    events = pd.read_csv(events_path, sep="\t")
    if not {"onset", "duration", "material", "response"}.issubset(events.columns):
        raise ValueError(f"事件表缺少必要列: {events_path}")

    mask = np.asarray(mask_img.dataobj) > 0
    data = np.asarray(img.dataobj[..., keep_slice], dtype=np.float32)
    y = data[mask, :].T.astype(np.float32)
    del data

    # 转成百分比信号变化，便于不同 run/subject 的 beta 量纲更接近。
    mean_signal = np.mean(y, axis=0, keepdims=True)
    valid_voxels = np.squeeze(mean_signal > 1e-6)
    y[:, valid_voxels] = 100.0 * (y[:, valid_voxels] / mean_signal[:, valid_voxels] - 1.0)
    y[:, ~valid_voxels] = 0.0
    y = np.nan_to_num(y, copy=False)

    confounds = select_confounds(confounds_path, cfg)
    nuisance = confounds.iloc[keep_slice, :].to_numpy(dtype=np.float64)
    nuisance = standardize_columns(nuisance)
    drift = make_dct_drift(kept_n, tr, float(cfg.get("high_pass_hz", 0.008)))
    nuisance = np.column_stack([nuisance, drift]) if nuisance.size else drift
    nuisance = standardize_columns(nuisance)
    intercept = np.ones((kept_n, 1), dtype=np.float64)

    all_onsets = events["onset"].to_numpy(dtype=float)
    all_durations = events["duration"].to_numpy(dtype=float)
    all_regressors = np.column_stack(
        [
            events_to_regressor(np.array([on]), np.array([dur]), n_scans, tr)[keep_slice]
            for on, dur in zip(all_onsets, all_durations)
        ]
    )

    beta_matrix = np.zeros((len(events), int(mask.sum())), dtype=np.float32)
    for trial_idx in range(len(events)):
        target = all_regressors[:, [trial_idx]]
        other = np.sum(np.delete(all_regressors, trial_idx, axis=1), axis=1, keepdims=True)
        design = np.column_stack([target, other, nuisance, intercept])
        design = np.nan_to_num(design)
        beta_matrix[trial_idx, :] = (np.linalg.pinv(design)[0, :] @ y).astype(np.float32)

    beta_4d = np.zeros(mask.shape + (len(events),), dtype=np.float32)
    beta_4d[mask, :] = beta_matrix.T
    out_img = nib.Nifti1Image(beta_4d, affine=img.affine, header=img.header)
    out_img.header.set_data_dtype(np.float32)
    out_img.header["dim"][0] = 4
    out_img.header["dim"][4] = len(events)
    nib.save(out_img, str(beta_path))

    meta = events.copy()
    meta.insert(0, "trial_id", [f"{sub}_run-{run}_trial-{i + 1:03d}" for i in range(len(meta))])
    meta.insert(1, "subject", sub)
    meta.insert(2, "run", run)
    meta["beta_index"] = np.arange(len(meta))
    meta["include_mvpa"] = meta["response"].astype(str) != str(cfg.get("exclude_response_value_from_mvpa", 1))
    meta["source_events"] = str(events_path)
    meta["source_preproc_bold"] = str(preproc_path)
    meta.to_csv(meta_path, sep="\t", index=False)

    design_summary = {
        "subject": sub,
        "run": run,
        "tr": tr,
        "n_scans_original": n_scans,
        "trim_start_tr": trim_start,
        "trim_end_tr": trim_end,
        "n_scans_model": kept_n,
        "n_trials": int(len(events)),
        "n_voxels_in_mask": int(mask.sum()),
        "n_valid_mean_signal_voxels": int(valid_voxels.sum()),
        "n_nuisance_columns_after_standardization": int(nuisance.shape[1]) if nuisance.size else 0,
        "confounds_used": list(confounds.columns),
        "high_pass_hz": float(cfg.get("high_pass_hz", 0.008)),
        "beta_file": str(beta_path),
        "metadata_file": str(meta_path),
    }
    with design_path.open("w", encoding="utf-8") as f:
        json.dump(design_summary, f, ensure_ascii=False, indent=2)

    print(f"完成 LSS beta: {beta_path}")
    return beta_path, meta_path


def main() -> None:
    parser = argparse.ArgumentParser(description="提取 trial-level LSS beta。")
    parser.add_argument("--config", type=Path, default=Path("analysis/config/analysis_config.json"))
    parser.add_argument("--subject", default=None, help="例如 sub-001；默认处理所有已完成 run。")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    cfg = read_config(args.config if args.config.exists() else None)
    bids_dir = Path(cfg.get("bids_dir", "data/bids_dataset"))
    fmriprep_dir = Path(cfg.get("fmriprep_dir", "data/derivatives"))
    analysis_dir = Path(cfg.get("analysis_dir", "data/derivatives/tactile_lss_mvpa_rsa"))
    out_dir = analysis_dir / "lss"
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = iter_runs(
        bids_dir=bids_dir,
        fmriprep_dir=fmriprep_dir,
        task=str(cfg.get("task", "tactile")),
        space=str(cfg.get("space", "T1w")),
        subject=args.subject,
    )
    if not runs:
        raise SystemExit("没有找到可用于 LSS 的完整 fMRIPrep run。请先补齐 derivatives。")

    produced = []
    for run_info in runs:
        produced.append(fit_lss_for_run(run_info, cfg, out_dir, args.overwrite))

    manifest = pd.DataFrame(
        [{"beta_file": str(beta), "metadata_file": str(meta)} for beta, meta in produced]
    )
    manifest.to_csv(out_dir / "lss_outputs.tsv", sep="\t", index=False)
    print(f"\nLSS 输出清单: {out_dir / 'lss_outputs.tsv'}")


if __name__ == "__main__":
    main()
