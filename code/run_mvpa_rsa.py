#!/usr/bin/env python
"""基于 trial beta 进行全脑/ROI MVPA 与 RSA，并计算 Top-k 排序。

默认策略：
- MVPA：leave-one-run-out，LinearSVC 决策分数排序，统计真实 material 是否进入 Top1/Top5/Top10。
- RSA：leave-one-run-out，用训练 run 的 material 原型与测试 trial 做相关相似度检索。
- ROI：读取 analysis/rois 中与 beta 空间一致的 NIfTI mask。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC


def read_config(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def beta_path_from_meta(meta_path: Path) -> Path:
    return Path(str(meta_path).replace("_events.tsv", "_beta.nii.gz"))


def find_lss_outputs(lss_dir: Path, subject: str | None) -> list[tuple[Path, Path]]:
    pattern = f"{subject or 'sub-*'}/func/*_desc-lssTrialBetas_events.tsv"
    pairs = []
    for meta_path in sorted(lss_dir.glob(pattern)):
        beta_path = beta_path_from_meta(meta_path)
        if beta_path.exists():
            pairs.append((beta_path, meta_path))
    return pairs


def load_metadata(pairs: list[tuple[Path, Path]], label_col: str, exclude_response: bool) -> pd.DataFrame:
    rows = []
    for beta_path, meta_path in pairs:
        df = pd.read_csv(meta_path, sep="\t")
        df["beta_file"] = str(beta_path)
        df["metadata_file"] = str(meta_path)
        rows.append(df)
    meta = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if meta.empty:
        return meta
    if label_col in {"material_initial", "material_letter", "letter_class"} and "material" in meta.columns:
        # 中文说明：把 P3、P4、P6 等同一首字母的 material 合并成一类。
        # 这样可以把 80 个细分类降到 13 个字母类别，提高每类训练样本数。
        meta[label_col] = (
            meta["material"]
            .astype(str)
            .str.extract(r"^([A-Za-z])", expand=False)
            .str.upper()
        )
    if label_col not in meta.columns:
        raise ValueError(f"trial 元数据缺少标签列: {label_col}")
    if exclude_response and "include_mvpa" in meta.columns:
        meta = meta[meta["include_mvpa"].astype(str).str.lower().isin(["true", "1"])].copy()
    meta[label_col] = meta[label_col].astype(str)
    meta["run"] = meta["run"].astype(str).str.zfill(2)
    return meta.reset_index(drop=True)


def load_beta_matrix(meta: pd.DataFrame, mask: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """读取 beta 并返回 trial x voxel 矩阵、实际 mask、参考图像。"""
    xs = []
    actual_mask = mask
    ref_img = None
    for beta_file, group in meta.groupby("beta_file", sort=False):
        img = nib.load(beta_file)
        if ref_img is None:
            ref_img = img
        data = np.asarray(img.dataobj, dtype=np.float32)
        if actual_mask is None:
            # LSS beta 的脑外体素为 0。用任一 trial 非零体素作为全脑特征 mask。
            actual_mask = np.any(np.abs(data) > 1e-8, axis=3)
        idx = group["beta_index"].to_numpy(dtype=int)
        x = data[..., idx][actual_mask, :].T
        xs.append(x.astype(np.float32))
    if ref_img is None or actual_mask is None:
        raise ValueError("没有可读取的 beta 图像。")
    x_all = np.vstack(xs)
    x_all = np.nan_to_num(x_all, copy=False)
    return x_all, actual_mask, ref_img


def limit_voxels_by_variance(x: np.ndarray, max_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    """全脑特征很多时，先按 trial 间方差保留前 N 个体素。"""
    if max_voxels <= 0 or x.shape[1] <= max_voxels:
        return x, np.arange(x.shape[1])
    var = np.nanvar(x, axis=0)
    keep = np.argsort(var)[-max_voxels:]
    keep.sort()
    return x[:, keep], keep


def topk_from_scores(scores: np.ndarray, classes: np.ndarray, y_true: np.ndarray, topk: list[int]) -> tuple[dict[str, float], pd.DataFrame]:
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    order = np.argsort(scores, axis=1)[:, ::-1]
    rows = []
    hits = {k: [] for k in topk}
    for i, true_label in enumerate(y_true):
        ranked_classes = classes[order[i]]
        ranked_scores = scores[i, order[i]]
        rank_pos = np.where(ranked_classes == true_label)[0]
        true_rank = int(rank_pos[0] + 1) if len(rank_pos) else math.inf
        for k in topk:
            hits[k].append(true_rank <= k)
        row: dict[str, Any] = {"true_label": true_label, "true_rank": true_rank}
        for j in range(min(10, len(ranked_classes))):
            row[f"rank{j + 1}_label"] = ranked_classes[j]
            row[f"rank{j + 1}_score"] = float(ranked_scores[j])
        rows.append(row)
    metrics = {f"top{k}_acc": float(np.mean(hits[k])) if hits[k] else np.nan for k in topk}
    metrics["n_tested"] = int(len(y_true))
    return metrics, pd.DataFrame(rows)


def augment_training_data(
    x_train: np.ndarray,
    y_train: np.ndarray,
    augmentation_cfg: dict[str, Any] | None,
    fold_seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """只增强训练集，避免测试集信息泄漏。

    增强策略：
    1. 在同一类别内随机抽取两个 trial beta 做 mixup；
    2. 给合成样本加入很小的高斯扰动；
    3. 原始训练样本保留，合成样本只追加到训练集。
    """
    cfg = augmentation_cfg or {}
    if not bool(cfg.get("enabled", False)):
        return x_train, y_train, {"augmentation_enabled": False, "n_synthetic_train": 0}

    copies_per_sample = int(cfg.get("copies_per_sample", 2))
    mixup_alpha = float(cfg.get("mixup_alpha", 0.4))
    noise_sd_fraction = float(cfg.get("noise_sd_fraction", 0.01))
    if copies_per_sample <= 0:
        return x_train, y_train, {"augmentation_enabled": False, "n_synthetic_train": 0}

    rng = np.random.default_rng(int(cfg.get("random_state", 20260609)) + fold_seed)
    feature_sd = np.nanstd(x_train, axis=0).astype(np.float32)
    feature_sd[feature_sd < 1e-8] = 0.0

    synthetic_x = []
    synthetic_y = []
    for label in sorted(np.unique(y_train)):
        idx = np.where(y_train == label)[0]
        if len(idx) < 2:
            continue
        n_new = len(idx) * copies_per_sample
        a = rng.choice(idx, size=n_new, replace=True)
        b = rng.choice(idx, size=n_new, replace=True)
        if mixup_alpha > 0:
            lam = rng.beta(mixup_alpha, mixup_alpha, size=(n_new, 1)).astype(np.float32)
        else:
            lam = np.full((n_new, 1), 0.5, dtype=np.float32)
        mixed = lam * x_train[a] + (1.0 - lam) * x_train[b]
        if noise_sd_fraction > 0:
            noise = rng.normal(0.0, noise_sd_fraction, size=mixed.shape).astype(np.float32)
            mixed = mixed + noise * feature_sd
        synthetic_x.append(mixed.astype(np.float32))
        synthetic_y.extend([label] * n_new)

    if not synthetic_x:
        return x_train, y_train, {"augmentation_enabled": False, "n_synthetic_train": 0}

    x_aug = np.vstack([x_train, *synthetic_x]).astype(np.float32)
    y_aug = np.concatenate([y_train, np.asarray(synthetic_y, dtype=y_train.dtype)])
    info = {
        "augmentation_enabled": True,
        "n_synthetic_train": int(len(synthetic_y)),
        "n_train_after_augmentation": int(len(y_aug)),
        "augmentation_copies_per_sample": copies_per_sample,
        "augmentation_mixup_alpha": mixup_alpha,
        "augmentation_noise_sd_fraction": noise_sd_fraction,
    }
    return x_aug, y_aug, info


def evaluate_mvpa(
    x: np.ndarray,
    y: np.ndarray,
    runs: np.ndarray,
    topk: list[int],
    feature_k: int,
    augmentation_cfg: dict[str, Any] | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    """leave-one-run-out 多分类 MVPA。"""
    all_rank_rows = []
    fold_metrics = []
    for test_run in sorted(np.unique(runs)):
        train = runs != test_run
        test = runs == test_run
        train_classes = set(y[train])
        test = test & np.array([label in train_classes for label in y])
        if test.sum() == 0 or len(np.unique(y[train])) < 2:
            continue
        x_train = x[train]
        y_train = y[train]
        x_train, y_train, augmentation_info = augment_training_data(
            x_train,
            y_train,
            augmentation_cfg,
            fold_seed=int(test_run) if str(test_run).isdigit() else len(fold_metrics),
        )
        k = min(feature_k, x.shape[1])
        steps: list[Any] = [StandardScaler()]
        if k > 0 and x.shape[1] > k:
            steps.append(SelectKBest(f_classif, k=k))
        classifier_cfg = augmentation_cfg or {}
        svm_c = float(classifier_cfg.get("svm_c", 1.0))
        svm_max_iter = int(classifier_cfg.get("svm_max_iter", 20000))
        steps.append(LinearSVC(C=svm_c, class_weight="balanced", max_iter=svm_max_iter))
        clf = make_pipeline(*steps)
        clf.fit(x_train, y_train)
        scores = clf.decision_function(x[test])
        classes = clf[-1].classes_
        metrics, ranks = topk_from_scores(scores, classes, y[test], topk)
        metrics["fold_run"] = test_run
        metrics["n_train"] = int(train.sum())
        metrics.update(augmentation_info)
        fold_metrics.append(metrics)
        ranks.insert(0, "fold_run", test_run)
        ranks.insert(1, "test_index", np.where(test)[0])
        all_rank_rows.append(ranks)

    if not fold_metrics:
        return {"status": "skipped", "reason": "少于 2 个可交叉验证的 run 或训练集中缺少测试标签。"}, pd.DataFrame()

    fold_df = pd.DataFrame(fold_metrics)
    summary: dict[str, Any] = {"status": "ok", "n_folds": int(len(fold_df))}
    for k in topk:
        col = f"top{k}_acc"
        summary[col] = float(np.average(fold_df[col], weights=fold_df["n_tested"]))
    summary["n_tested"] = int(fold_df["n_tested"].sum())
    if "augmentation_enabled" in fold_df.columns:
        summary["augmentation_enabled"] = bool(fold_df["augmentation_enabled"].any())
    if "n_synthetic_train" in fold_df.columns:
        summary["mean_n_synthetic_train_per_fold"] = float(fold_df["n_synthetic_train"].mean())
    if "n_train_after_augmentation" in fold_df.columns:
        summary["mean_n_train_after_augmentation"] = float(fold_df["n_train_after_augmentation"].mean())
    return summary, pd.concat(all_rank_rows, ignore_index=True)


def row_normalize(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    norm[norm < 1e-8] = 1.0
    return x / norm


def evaluate_rsa_retrieval(x: np.ndarray, y: np.ndarray, runs: np.ndarray, topk: list[int]) -> tuple[dict[str, Any], pd.DataFrame]:
    """用训练 run 的类别原型做 RSA 检索，并计算 Top-k。"""
    all_rank_rows = []
    fold_metrics = []
    for test_run in sorted(np.unique(runs)):
        train = runs != test_run
        test = runs == test_run
        train_classes = np.array(sorted(set(y[train])))
        test = test & np.array([label in set(train_classes) for label in y])
        if test.sum() == 0 or len(train_classes) < 2:
            continue
        scaler = StandardScaler()
        x_train = scaler.fit_transform(x[train])
        x_test = scaler.transform(x[test])
        prototypes = np.vstack([x_train[y[train] == label].mean(axis=0) for label in train_classes])
        scores = row_normalize(x_test) @ row_normalize(prototypes).T
        metrics, ranks = topk_from_scores(scores, train_classes, y[test], topk)
        metrics["fold_run"] = test_run
        metrics["n_train"] = int(train.sum())
        fold_metrics.append(metrics)
        ranks.insert(0, "fold_run", test_run)
        ranks.insert(1, "test_index", np.where(test)[0])
        all_rank_rows.append(ranks)

    if not fold_metrics:
        return {"status": "skipped", "reason": "少于 2 个可交叉验证的 run 或训练集中缺少测试标签。"}, pd.DataFrame()

    fold_df = pd.DataFrame(fold_metrics)
    summary: dict[str, Any] = {"status": "ok", "n_folds": int(len(fold_df))}
    for k in topk:
        col = f"top{k}_acc"
        summary[col] = float(np.average(fold_df[col], weights=fold_df["n_tested"]))
    summary["n_tested"] = int(fold_df["n_tested"].sum())
    return summary, pd.concat(all_rank_rows, ignore_index=True)


def load_roi_masks(roi_dir: Path, ref_img: nib.Nifti1Image) -> list[tuple[str, np.ndarray, str]]:
    """读取与 beta 图像同空间的 ROI mask。"""
    masks = []
    if not roi_dir.exists():
        return masks
    for path in sorted(list(roi_dir.glob("*.nii")) + list(roi_dir.glob("*.nii.gz"))):
        name = path.name.replace(".nii.gz", "").replace(".nii", "")
        img = nib.load(str(path))
        if img.shape[:3] != ref_img.shape[:3] or not np.allclose(img.affine, ref_img.affine, atol=1e-3):
            masks.append((name, np.array([], dtype=bool), f"ROI 与 beta 空间不一致，已跳过: {path}"))
            continue
        mask = np.asarray(img.dataobj) > 0
        if int(mask.sum()) < 2:
            masks.append((name, np.array([], dtype=bool), f"ROI 体素少于 2，已跳过: {path}"))
            continue
        masks.append((name, mask, ""))
    return masks


def write_ranked_region_tables(results: pd.DataFrame, out_dir: Path, topk: list[int]) -> None:
    """把 ROI/全脑结果按 Top1/Top5/Top10 分别排序。"""
    if results.empty:
        return
    ranked = results.copy()
    sort_cols = [f"top{k}_acc" for k in topk if f"top{k}_acc" in ranked.columns]
    if sort_cols:
        ranked = ranked.sort_values(sort_cols, ascending=[False] * len(sort_cols))
    ranked.to_csv(out_dir / "ranked_results_all.tsv", sep="\t", index=False)
    for k in topk:
        col = f"top{k}_acc"
        if col in results.columns:
            tmp = results.sort_values(col, ascending=False).copy()
            tmp.insert(0, f"rank_by_top{k}", np.arange(1, len(tmp) + 1))
            tmp.to_csv(out_dir / f"ranked_results_top{k}.tsv", sep="\t", index=False)


def sphere_offsets(radius: int) -> np.ndarray:
    coords = []
    for x in range(-radius, radius + 1):
        for y in range(-radius, radius + 1):
            for z in range(-radius, radius + 1):
                if x * x + y * y + z * z <= radius * radius:
                    coords.append((x, y, z))
    return np.asarray(coords, dtype=int)


def run_sparse_searchlight(
    x_full: np.ndarray,
    full_mask: np.ndarray,
    ref_img: nib.Nifti1Image,
    y: np.ndarray,
    runs: np.ndarray,
    out_dir: Path,
    topk: list[int],
    feature_k: int,
    radius: int,
    step: int,
) -> pd.DataFrame:
    """稀疏 searchlight，输出 MVPA/RSA Top-k NIfTI 图。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    voxel_to_col = np.full(full_mask.shape, -1, dtype=int)
    voxel_to_col[full_mask] = np.arange(int(full_mask.sum()))
    centers = np.argwhere(full_mask)
    centers = centers[(centers[:, 0] % step == 0) & (centers[:, 1] % step == 0) & (centers[:, 2] % step == 0)]
    offsets = sphere_offsets(radius)

    maps: dict[str, np.ndarray] = {}
    for analysis in ["mvpa", "rsa"]:
        for k in topk:
            maps[f"{analysis}_top{k}"] = np.full(full_mask.shape, np.nan, dtype=np.float32)

    rows = []
    shape = np.array(full_mask.shape)
    for i, center in enumerate(centers):
        sphere = center + offsets
        valid = np.all((sphere >= 0) & (sphere < shape), axis=1)
        sphere = sphere[valid]
        cols = voxel_to_col[sphere[:, 0], sphere[:, 1], sphere[:, 2]]
        cols = cols[cols >= 0]
        if len(cols) < 5:
            continue
        x = x_full[:, cols]
        mvpa_summary, _ = evaluate_mvpa(x, y, runs, topk, min(feature_k, x.shape[1]))
        rsa_summary, _ = evaluate_rsa_retrieval(x, y, runs, topk)
        row: dict[str, Any] = {
            "center_x": int(center[0]),
            "center_y": int(center[1]),
            "center_z": int(center[2]),
            "n_voxels": int(len(cols)),
            "mvpa_status": mvpa_summary.get("status"),
            "rsa_status": rsa_summary.get("status"),
        }
        for analysis, summary in [("mvpa", mvpa_summary), ("rsa", rsa_summary)]:
            for k in topk:
                col = f"top{k}_acc"
                value = summary.get(col, np.nan)
                row[f"{analysis}_{col}"] = value
                maps[f"{analysis}_top{k}"][tuple(center)] = value
        rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"searchlight 已处理 {i + 1}/{len(centers)} 个中心点")

    for name, data in maps.items():
        img = nib.Nifti1Image(data, affine=ref_img.affine, header=ref_img.header)
        img.header.set_data_dtype(np.float32)
        nib.save(img, str(out_dir / f"desc-searchlight_{name}.nii.gz"))
    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "searchlight_centers.tsv", sep="\t", index=False)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="运行全脑/ROI MVPA 与 RSA，并输出 Top-k 排序。")
    parser.add_argument("--config", type=Path, default=Path("analysis/config/analysis_config.json"))
    parser.add_argument("--subject", default=None, help="例如 sub-001；默认处理所有有 LSS beta 的受试者。")
    parser.add_argument("--include-response1", action="store_true", help="默认排除 response=1 的 1-back 重复 trial。")
    parser.add_argument("--run-searchlight", action="store_true", help="启用稀疏 searchlight 图，计算较慢。")
    args = parser.parse_args()

    cfg = read_config(args.config if args.config.exists() else None)
    analysis_dir = Path(cfg.get("analysis_dir", "data/derivatives/tactile_lss_mvpa_rsa"))
    lss_dir = Path(cfg.get("lss_dir", str(analysis_dir / "lss")))
    out_root = Path(cfg.get("mvpa_rsa_dir", str(analysis_dir / "mvpa_rsa")))
    out_root.mkdir(parents=True, exist_ok=True)
    label_col = str(cfg.get("trial_label_column", "material"))
    topk = [int(k) for k in cfg.get("topk", [1, 5, 10])]
    feature_k = int(cfg.get("mvpa_feature_k", 5000))
    wholebrain_max_voxels = int(cfg.get("wholebrain_max_voxels", 30000))
    augmentation_cfg = cfg.get("mvpa_augmentation", {})

    subjects = [args.subject] if args.subject else sorted({p.parts[-3] for p in lss_dir.glob("sub-*/func/*_events.tsv")})
    if not subjects:
        raise SystemExit("没有找到 LSS beta 元数据。请先运行 run_lss_glm.py。")

    all_results = []
    for subject in subjects:
        pairs = find_lss_outputs(lss_dir, subject)
        subject_out = out_root / subject
        rank_out = subject_out / "ranking"
        report_out = subject_out / "reports"
        rank_out.mkdir(parents=True, exist_ok=True)
        report_out.mkdir(parents=True, exist_ok=True)

        if not pairs:
            reason = "没有找到 LSS beta 输出；请先运行 run_lss_glm.py，且 fMRIPrep run 必须完整。"
            (report_out / "skip_reason.txt").write_text(reason + "\n", encoding="utf-8")
            all_results.append(
                {
                    "subject": subject,
                    "analysis": "all",
                    "region": "all",
                    "status": "skipped",
                    "reason": reason,
                    "n_trials": 0,
                    "n_runs": 0,
                    "n_features": 0,
                }
            )
            continue

        meta = load_metadata(pairs, label_col, exclude_response=not args.include_response1)
        if meta.empty:
            reason = "没有可用于 MVPA/RSA 的 trial。"
            (report_out / "skip_reason.txt").write_text(reason + "\n", encoding="utf-8")
            all_results.append(
                {
                    "subject": subject,
                    "analysis": "all",
                    "region": "all",
                    "status": "skipped",
                    "reason": reason,
                    "n_trials": 0,
                    "n_runs": 0,
                    "n_features": 0,
                }
            )
            continue
        meta.to_csv(report_out / "mvpa_rsa_trials.tsv", sep="\t", index=False)

        x_full, full_mask, ref_img = load_beta_matrix(meta)
        y = meta[label_col].to_numpy(dtype=str)
        runs = meta["run"].to_numpy(dtype=str)
        if len(np.unique(runs)) < 2:
            reason = "当前只有 1 个完整 run，无法做 leave-one-run-out MVPA/RSA。"
            (report_out / "skip_reason.txt").write_text(reason + "\n", encoding="utf-8")
            all_results.append(
                {
                    "subject": subject,
                    "analysis": "wholebrain",
                    "region": "wholebrain",
                    "status": "skipped",
                    "reason": reason,
                    "n_trials": int(len(meta)),
                    "n_runs": int(len(np.unique(runs))),
                    "n_features": int(x_full.shape[1]),
                }
            )
            continue

        x_whole, kept_cols = limit_voxels_by_variance(x_full, wholebrain_max_voxels)
        for analysis_name, evaluator in [("wholebrain_mvpa", evaluate_mvpa), ("wholebrain_rsa", evaluate_rsa_retrieval)]:
            if analysis_name.endswith("mvpa"):
                summary, ranks = evaluator(x_whole, y, runs, topk, feature_k, augmentation_cfg)
            else:
                summary, ranks = evaluator(x_whole, y, runs, topk)
            summary.update(
                {
                    "subject": subject,
                    "analysis": analysis_name,
                    "region": "wholebrain",
                    "n_trials": int(len(meta)),
                    "n_runs": int(len(np.unique(runs))),
                    "n_features": int(x_whole.shape[1]),
                }
            )
            all_results.append(summary)
            if not ranks.empty:
                ranks.to_csv(rank_out / f"{analysis_name}_trial_ranks.tsv", sep="\t", index=False)

        roi_rows = []
        for roi_name, roi_mask, problem in load_roi_masks(Path(cfg.get("roi_dir", "analysis/rois")), ref_img):
            if problem:
                roi_rows.append({"roi": roi_name, "status": "skipped", "reason": problem})
                continue
            x_roi, _, _ = load_beta_matrix(meta, mask=roi_mask)
            for analysis_name, evaluator in [("roi_mvpa", evaluate_mvpa), ("roi_rsa", evaluate_rsa_retrieval)]:
                if analysis_name.endswith("mvpa"):
                    summary, ranks = evaluator(x_roi, y, runs, topk, min(feature_k, x_roi.shape[1]), augmentation_cfg)
                else:
                    summary, ranks = evaluator(x_roi, y, runs, topk)
                summary.update(
                    {
                        "subject": subject,
                        "analysis": analysis_name,
                        "region": roi_name,
                        "n_trials": int(len(meta)),
                        "n_runs": int(len(np.unique(runs))),
                        "n_features": int(x_roi.shape[1]),
                    }
                )
                all_results.append(summary)
                roi_rows.append(summary.copy())
                if not ranks.empty:
                    ranks.to_csv(rank_out / f"{analysis_name}_{roi_name}_trial_ranks.tsv", sep="\t", index=False)
        if roi_rows:
            pd.DataFrame(roi_rows).to_csv(report_out / "roi_status_and_results.tsv", sep="\t", index=False)

        if args.run_searchlight or bool(cfg.get("run_searchlight_by_default", False)):
            searchlight_out = subject_out / "searchlight"
            searchlight_df = run_sparse_searchlight(
                x_full=x_full,
                full_mask=full_mask,
                ref_img=ref_img,
                y=y,
                runs=runs,
                out_dir=searchlight_out,
                topk=topk,
                feature_k=feature_k,
                radius=int(cfg.get("searchlight_radius_voxels", 3)),
                step=int(cfg.get("searchlight_step_voxels", 3)),
            )
            if not searchlight_df.empty:
                for metric in [f"mvpa_top{k}_acc" for k in topk] + [f"rsa_top{k}_acc" for k in topk]:
                    if metric in searchlight_df.columns:
                        best = searchlight_df.sort_values(metric, ascending=False).head(10)
                        best.to_csv(searchlight_out / f"top10_centers_by_{metric}.tsv", sep="\t", index=False)

    results = pd.DataFrame(all_results)
    summary_path = out_root / "mvpa_rsa_topk_summary.tsv"
    if results.empty:
        results = pd.DataFrame(
            columns=["subject", "analysis", "region", "status", "reason", "n_trials", "n_runs", "n_features"]
        )
    results.to_csv(summary_path, sep="\t", index=False)
    ok_results = results[results["status"] == "ok"] if "status" in results.columns else results
    write_ranked_region_tables(ok_results, out_root, topk)
    print(f"MVPA/RSA 汇总: {summary_path}")


if __name__ == "__main__":
    main()
