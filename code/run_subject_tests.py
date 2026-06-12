#!/usr/bin/env python
"""单被试和跨被试测试脚本 - 使用真实 LSS beta 数据"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import joblib
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# 数据加载（参考 run_mvpa_rsa.py）
# ---------------------------------------------------------------------------

def beta_path_from_meta(meta_path: Path) -> Path:
    """从元数据路径推断 beta NIfTI 路径"""
    return Path(str(meta_path).replace("_events.tsv", "_beta.nii.gz"))


def find_lss_outputs(lss_dir: Path, subject: str | None = None) -> list[tuple[Path, Path]]:
    """查找 LSS beta 输出，返回 (beta_path, meta_path) 对"""
    pattern = f"{subject or 'sub-*'}/func/*_desc-lssTrialBetas_events.tsv"
    pairs = []
    for meta_path in sorted(lss_dir.glob(pattern)):
        beta_path = beta_path_from_meta(meta_path)
        if beta_path.exists():
            pairs.append((beta_path, meta_path))
    return pairs


def find_lss_events(lss_dir: Path, subject: str | None = None) -> list[Path]:
    """查找 LSS 事件文件（即使 beta 不存在也返回）"""
    pattern = f"{subject or 'sub-*'}/func/*_desc-lssTrialBetas_events.tsv"
    return sorted(lss_dir.glob(pattern))


def load_metadata(pairs: list[tuple[Path, Path]], label_col: str = "material",
                  exclude_response: bool = True) -> pd.DataFrame:
    """加载元数据，提取字母类别标签"""
    rows = []
    for beta_path, meta_path in pairs:
        df = pd.read_csv(meta_path, sep="\t")
        df["beta_file"] = str(beta_path)
        df["metadata_file"] = str(meta_path)
        rows.append(df)
    meta = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    if meta.empty:
        return meta
    # 提取首字母作为类别标签
    if "material" in meta.columns:
        meta["letter_class"] = (
            meta["material"]
            .astype(str)
            .str.extract(r"^([A-Za-z])", expand=False)
            .str.upper()
        )
    if label_col not in meta.columns and "letter_class" in meta.columns:
        label_col = "letter_class"
    if exclude_response and "include_mvpa" in meta.columns:
        meta = meta[meta["include_mvpa"].astype(str).str.lower().isin(["true", "1"])].copy()
    meta["letter_class"] = meta["letter_class"].astype(str)
    meta["run"] = meta["run"].astype(str).str.zfill(2)
    return meta.reset_index(drop=True)


def load_beta_matrix(meta: pd.DataFrame, mask: np.ndarray | None = None
                     ) -> tuple[np.ndarray, np.ndarray, nib.Nifti1Image]:
    """读取 beta 并返回 trial x voxel 矩阵、实际 mask、参考图像。
    
    不同 run 的 beta 文件可能有不同的体素数，这里统一使用
    每个文件各自的非零体素 mask，然后截取到最小公共体素数。
    """
    xs = []
    ref_img = None
    for beta_file, group in meta.groupby("beta_file", sort=False):
        img = nib.load(beta_file)
        if ref_img is None:
            ref_img = img
        data = np.asarray(img.dataobj, dtype=np.float32)
        idx = group["beta_index"].to_numpy(dtype=int)
        # 每个 beta 文件使用自己的非零 mask
        file_mask = np.any(np.abs(data) > 1e-8, axis=3)
        x = data[..., idx][file_mask, :].T
        xs.append(x.astype(np.float32))
    if ref_img is None:
        raise ValueError("没有可读取的 beta 图像。")
    # 统一特征维度：取最小公共体素数
    min_features = min(x.shape[1] for x in xs)
    xs = [x[:, :min_features] for x in xs]
    x_all = np.vstack(xs)
    x_all = np.nan_to_num(x_all, copy=False)
    # 返回一个虚拟 mask（仅用于接口兼容）
    actual_mask = np.ones(ref_img.shape[:3], dtype=bool)
    return x_all, actual_mask, ref_img


def limit_voxels_by_variance(x: np.ndarray, max_voxels: int) -> tuple[np.ndarray, np.ndarray]:
    """按方差保留前 N 个体素"""
    if max_voxels <= 0 or x.shape[1] <= max_voxels:
        return x, np.arange(x.shape[1])
    var = np.nanvar(x, axis=0)
    keep = np.argsort(var)[-max_voxels:]
    keep.sort()
    return x[:, keep], keep


# ---------------------------------------------------------------------------
# Top-k 计算（参考 run_mvpa_rsa.py 的 topk_from_scores）
# ---------------------------------------------------------------------------

def topk_from_scores(scores: np.ndarray, classes: np.ndarray,
                     y_true: np.ndarray, topk: list[int]) -> dict[str, float]:
    """计算 Top-k 准确率"""
    if scores.ndim == 1:
        scores = np.column_stack([-scores, scores])
    order = np.argsort(scores, axis=1)[:, ::-1]
    hits = {k: [] for k in topk}
    for i, true_label in enumerate(y_true):
        ranked_classes = classes[order[i]]
        rank_pos = np.where(ranked_classes == true_label)[0]
        true_rank = int(rank_pos[0] + 1) if len(rank_pos) else 999
        for k in topk:
            hits[k].append(true_rank <= k)
    metrics = {f"top{k}_acc": float(np.mean(hits[k])) for k in topk}
    metrics["n_tested"] = int(len(y_true))
    return metrics


# ---------------------------------------------------------------------------
# 单被试测试（Leave-One-Run-Out）
# ---------------------------------------------------------------------------

def train_single_subject(lss_dir: Path, subject_id: str,
                         feature_k: int = 5000, max_voxels: int = 30000,
                         svm_c: float = 1.0) -> dict[str, Any]:
    """单被试 Leave-One-Run-Out 交叉验证"""
    print(f"\n=== 单被试测试: sub-{subject_id.zfill(3)} ===")

    pairs = find_lss_outputs(lss_dir, f"sub-{subject_id.zfill(3)}")
    if not pairs:
        # 检查是否有 events 文件但缺少 beta
        events = find_lss_events(lss_dir, f"sub-{subject_id.zfill(3)}")
        if events:
            print(f"  [错误] 找到 {len(events)} 个事件文件，但没有对应的 beta NIfTI 文件。")
            print(f"  需要先运行 run_lss_glm.py 生成 beta 文件。")
        else:
            print(f"  [错误] 未找到被试 sub-{subject_id.zfill(3)} 的数据。")
        return {"subject": subject_id, "status": "skipped"}

    meta = load_metadata(pairs, label_col="letter_class")
    if meta.empty:
        print(f"  [错误] 没有可用的 trial 数据。")
        return {"subject": subject_id, "status": "skipped"}

    x_full, full_mask, ref_img = load_beta_matrix(meta)
    x, _ = limit_voxels_by_variance(x_full, max_voxels)

    y = meta["letter_class"].to_numpy(dtype=str)
    runs = meta["run"].to_numpy(dtype=str)

    if len(np.unique(runs)) < 2:
        print(f"  [错误] 只有 1 个 run，无法做 Leave-One-Run-Out。")
        return {"subject": subject_id, "status": "skipped"}

    topk = [1, 3, 5]
    fold_results = []

    for test_run in sorted(np.unique(runs)):
        train_mask = runs != test_run
        test_mask = runs == test_run

        # 确保测试集中的标签在训练集中出现过
        train_classes = set(y[train_mask])
        valid_test = test_mask & np.array([label in train_classes for label in y])
        if valid_test.sum() == 0:
            continue

        x_train, y_train = x[train_mask], y[train_mask]
        x_test, y_test = x[valid_test], y[valid_test]

        # 构建管道：StandardScaler + SelectKBest + CalibratedClassifierCV(LinearSVC)
        k = min(feature_k, x.shape[1])
        base_svc = LinearSVC(C=svm_c, class_weight="balanced", max_iter=20000)
        steps = [StandardScaler()]
        if k > 0 and x.shape[1] > k:
            steps.append(SelectKBest(f_classif, k=k))
        steps.append(CalibratedClassifierCV(base_svc, cv=3))
        model = make_pipeline(*steps)
        model.fit(x_train, y_train)

        # 使用 predict_proba 获取概率
        probs = model.predict_proba(x_test)
        classes = model.named_steps["calibratedclassifiercv"].classes_

        metrics = topk_from_scores(probs, classes, y_test, topk)
        metrics["fold_run"] = test_run
        metrics["n_train"] = int(train_mask.sum())
        fold_results.append(metrics)

        print(f"  Run {test_run}: Top1={metrics['top1_acc']:.2%}, "
              f"Top3={metrics['top3_acc']:.2%}, Top5={metrics['top5_acc']:.2%}")

    if not fold_results:
        return {"subject": subject_id, "status": "skipped"}

    # 加权平均
    result = {"subject": subject_id, "status": "ok", "n_trials": int(len(meta)),
              "n_runs": int(len(np.unique(runs))), "n_features": int(x.shape[1]),
              "fold_results": fold_results}
    for k in topk:
        col = f"top{k}_acc"
        weights = [r["n_tested"] for r in fold_results]
        result[col] = float(np.average([r[col] for r in fold_results], weights=weights))

    print(f"  平均: Top1={result['top1_acc']:.2%}, "
          f"Top3={result['top3_acc']:.2%}, Top5={result['top5_acc']:.2%}")
    return result


# ---------------------------------------------------------------------------
# 跨被试测试（Leave-One-Subject-Out）
# ---------------------------------------------------------------------------

def train_cross_subject(lss_dir: Path, feature_k: int = 5000,
                        max_voxels: int = 30000, svm_c: float = 1.0
                        ) -> dict[str, Any]:
    """跨被试 Leave-One-Subject-Out 交叉验证"""
    print("\n=== 跨被试测试 (Leave-One-Subject-Out) ===")

    # 发现所有被试
    subjects = sorted({p.parts[-3] for p in find_lss_events(lss_dir)})
    if not subjects:
        print("  [错误] 未找到任何被试数据。")
        return {"type": "cross_subject", "status": "skipped"}

    topk = [1, 3, 5]
    subject_results = []

    for test_subject in subjects:
        print(f"\n  测试被试: {test_subject}")

        # 加载训练数据（排除测试被试）
        train_pairs = []
        for sub in subjects:
            if sub == test_subject:
                continue
            pairs = find_lss_outputs(lss_dir, sub)
            if not pairs:
                events = find_lss_events(lss_dir, sub)
                if events:
                    print(f"    [警告] {sub} 有事件文件但缺少 beta，已跳过。"
                          f"需要先运行 run_lss_glm.py 生成 beta 文件。")
                continue
            train_pairs.extend(pairs)

        # 加载测试数据
        test_pairs = find_lss_outputs(lss_dir, test_subject)
        if not test_pairs:
            events = find_lss_events(lss_dir, test_subject)
            if events:
                print(f"    [错误] {test_subject} 有事件文件但缺少 beta。"
                      f"需要先运行 run_lss_glm.py 生成 beta 文件。")
            else:
                print(f"    [错误] {test_subject} 无数据。")
            continue

        if not train_pairs:
            print(f"    [错误] 没有训练数据可用。")
            continue

        train_meta = load_metadata(train_pairs, label_col="letter_class")
        test_meta = load_metadata(test_pairs, label_col="letter_class")

        if train_meta.empty or test_meta.empty:
            print(f"    [错误] 训练或测试数据为空。")
            continue

        # 加载 beta 矩阵
        x_train_full, _, _ = load_beta_matrix(train_meta)
        x_test_full, _, _ = load_beta_matrix(test_meta)

        # 统一特征维度（取交集）
        min_features = min(x_train_full.shape[1], x_test_full.shape[1])
        x_train_full = x_train_full[:, :min_features]
        x_test_full = x_test_full[:, :min_features]

        # 按方差选择体素
        x_train, _ = limit_voxels_by_variance(x_train_full, max_voxels)
        x_test, _ = limit_voxels_by_variance(x_test_full, max_voxels)
        # 确保维度一致
        min_feat = min(x_train.shape[1], x_test.shape[1])
        x_train = x_train[:, :min_feat]
        x_test = x_test[:, :min_feat]

        y_train = train_meta["letter_class"].to_numpy(dtype=str)
        y_test = test_meta["letter_class"].to_numpy(dtype=str)

        # 确保测试标签在训练集中
        train_classes = set(y_train)
        valid_mask = np.array([label in train_classes for label in y_test])
        if valid_mask.sum() == 0:
            print(f"    [错误] 测试集中没有训练集包含的标签。")
            continue
        x_test = x_test[valid_mask]
        y_test = y_test[valid_mask]

        # 训练模型
        k = min(feature_k, x_train.shape[1])
        base_svc = LinearSVC(C=svm_c, class_weight="balanced", max_iter=20000)
        steps = [StandardScaler()]
        if k > 0 and x_train.shape[1] > k:
            steps.append(SelectKBest(f_classif, k=k))
        steps.append(CalibratedClassifierCV(base_svc, cv=3))
        model = make_pipeline(*steps)
        model.fit(x_train, y_train)

        probs = model.predict_proba(x_test)
        classes = model.named_steps["calibratedclassifiercv"].classes_

        metrics = topk_from_scores(probs, classes, y_test, topk)
        metrics["subject"] = test_subject
        metrics["n_train"] = int(len(y_train))
        subject_results.append(metrics)

        print(f"    Top1={metrics['top1_acc']:.2%}, "
              f"Top3={metrics['top3_acc']:.2%}, Top5={metrics['top5_acc']:.2%}")

    if not subject_results:
        return {"type": "cross_subject", "status": "skipped"}

    result = {"type": "cross_subject", "status": "ok",
              "n_subjects": len(subjects), "subject_results": subject_results}
    for k in topk:
        col = f"top{k}_acc"
        weights = [r["n_tested"] for r in subject_results]
        result[col] = float(np.average([r[col] for r in subject_results], weights=weights))

    print(f"\n  跨被试平均: Top1={result['top1_acc']:.2%}, "
          f"Top3={result['top3_acc']:.2%}, Top5={result['top5_acc']:.2%}")
    return result


# ---------------------------------------------------------------------------
# 保存最终模型（使用全部数据训练）
# ---------------------------------------------------------------------------

def save_final_model(lss_dir: Path, output_dir: Path,
                     feature_k: int = 5000, max_voxels: int = 30000,
                     svm_c: float = 1.0) -> None:
    """使用全部数据训练最终模型并保存"""
    print("\n=== 训练最终模型 ===")

    pairs = find_lss_outputs(lss_dir)
    if not pairs:
        print("  [错误] 没有找到 LSS beta 数据，无法训练最终模型。")
        return

    meta = load_metadata(pairs, label_col="letter_class")
    if meta.empty:
        print("  [错误] 没有可用的 trial 数据。")
        return

    x_full, _, _ = load_beta_matrix(meta)
    x, _ = limit_voxels_by_variance(x_full, max_voxels)
    y = meta["letter_class"].to_numpy(dtype=str)

    k = min(feature_k, x.shape[1])
    base_svc = LinearSVC(C=svm_c, class_weight="balanced", max_iter=20000)
    steps = [StandardScaler()]
    if k > 0 and x.shape[1] > k:
        steps.append(SelectKBest(f_classif, k=k))
    steps.append(CalibratedClassifierCV(base_svc, cv=3))
    model = make_pipeline(*steps)
    model.fit(x, y)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "mvpa_optimized.joblib"
    joblib.dump(model, model_path)
    print(f"  模型已保存到: {model_path}")

    # 保存模型信息
    classes = sorted(list(np.unique(y)))
    model_info = {
        "model_name": "Calibrated LinearSVC (LSS beta)",
        "n_train_samples": int(len(y)),
        "n_features": int(x.shape[1]),
        "feature_k": k,
        "svm_c": svm_c,
        "n_classes": len(classes),
        "classes": classes,
    }
    with open(output_dir / "mvpa_optimized.json", "w", encoding="utf-8") as f:
        json.dump(model_info, f, ensure_ascii=False, indent=2)

    # 保存测试集（取最后一个 run 的数据作为演示用测试集）
    runs = meta["run"].to_numpy(dtype=str)
    last_run = sorted(np.unique(runs))[-1]
    test_mask = runs == last_run
    x_test = x[test_mask]
    y_test = y[test_mask]

    test_set = {
        "x_test": x_test.tolist(),
        "y_test": y_test.tolist(),
    }
    with open(output_dir / "test_set.json", "w", encoding="utf-8") as f:
        json.dump(test_set, f)
    print(f"  测试集已保存到: {output_dir / 'test_set.json'}")


# ---------------------------------------------------------------------------
# 结果图表
# ---------------------------------------------------------------------------

def plot_results(single_results: list[dict], cross_result: dict, output_dir: Path) -> None:
    """生成结果图表"""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 过滤掉 skipped 的结果
    valid_single = [r for r in single_results if r.get("status") == "ok"]
    if not valid_single and cross_result.get("status") != "ok":
        print("  没有有效结果，跳过图表生成。")
        return

    subjects = [r["subject"] for r in valid_single]
    top1_scores = [r.get("top1_acc", 0) for r in valid_single]
    top3_scores = [r.get("top3_acc", 0) for r in valid_single]
    top5_scores = [r.get("top5_acc", 0) for r in valid_single]

    cross_top1 = cross_result.get("top1_acc", 0)
    cross_top3 = cross_result.get("top3_acc", 0)
    cross_top5 = cross_result.get("top5_acc", 0)

    fig, ax = plt.subplots(figsize=(10, 6))
    n = len(subjects) + (1 if cross_result.get("status") == "ok" else 0)
    x_pos = np.arange(n)
    width = 0.25

    all_top1 = top1_scores + ([cross_top1] if cross_result.get("status") == "ok" else [])
    all_top3 = top3_scores + ([cross_top3] if cross_result.get("status") == "ok" else [])
    all_top5 = top5_scores + ([cross_top5] if cross_result.get("status") == "ok" else [])

    ax.bar(x_pos - width, all_top1, width, label='Top-1', color='#1f77b4')
    ax.bar(x_pos, all_top3, width, label='Top-3', color='#ff7f0e')
    ax.bar(x_pos + width, all_top5, width, label='Top-5', color='#2ca02c')

    # 机会水平线
    chance_top1 = 1 / 13
    chance_top3 = 3 / 13
    chance_top5 = 5 / 13
    ax.axhline(y=chance_top1, color='#1f77b4', linestyle='--', alpha=0.5, label=f'Chance Top-1 ({chance_top1:.1%})')
    ax.axhline(y=chance_top3, color='#ff7f0e', linestyle='--', alpha=0.5, label=f'Chance Top-3 ({chance_top3:.1%})')
    ax.axhline(y=chance_top5, color='#2ca02c', linestyle='--', alpha=0.5, label=f'Chance Top-5 ({chance_top5:.1%})')

    labels = [f'S{sub}' for sub in subjects]
    if cross_result.get("status") == "ok":
        labels.append('跨被试')
    ax.set_xlabel('被试', fontsize=12)
    ax.set_ylabel('准确率', fontsize=12)
    ax.set_title('单被试与跨被试分类性能对比', fontsize=14)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, fontsize=10)
    ax.legend(fontsize=9)
    ax.set_ylim(0, 1.0)
    ax.grid(axis='y', alpha=0.3)

    for i, (t1, t3, t5) in enumerate(zip(all_top1, all_top3, all_top5)):
        ax.text(i - width, t1 + 0.01, f'{t1:.1%}', ha='center', fontsize=8)
        ax.text(i, t3 + 0.01, f'{t3:.1%}', ha='center', fontsize=8)
        ax.text(i + width, t5 + 0.01, f'{t5:.1%}', ha='center', fontsize=8)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\n结果图表已保存到: {output_dir / 'accuracy_comparison.png'}")


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------

def convert_numpy(obj):
    """递归转换 numpy 类型为 Python 原生类型"""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy(i) for i in obj]
    return obj


def main() -> None:
    parser = argparse.ArgumentParser(description="单被试和跨被试测试（使用真实 LSS beta 数据）")
    parser.add_argument("--data-dir", type=Path,
                        default=ROOT / "data" / "lss",
                        help="LSS beta 数据目录（默认: data/lss）")
    parser.add_argument("--output", type=Path, default=ROOT / "results",
                        help="结果输出目录")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "models",
                        help="模型保存目录")
    parser.add_argument("--feature-k", type=int, default=5000,
                        help="SelectKBest 保留的特征数")
    parser.add_argument("--max-voxels", type=int, default=30000,
                        help="按方差保留的最大体素数")
    parser.add_argument("--svm-c", type=float, default=1.0,
                        help="LinearSVC 的 C 参数")
    args = parser.parse_args()

    print("=" * 70)
    print("单被试和跨被试测试（使用真实 LSS beta 数据）")
    print(f"数据目录: {args.data_dir}")
    print("=" * 70)

    # 检查数据目录
    if not args.data_dir.exists():
        print(f"\n[错误] 数据目录不存在: {args.data_dir}")
        print("请确认数据路径，或使用 --data-dir 指定正确路径。")
        sys.exit(1)

    # 检查是否有 beta 文件
    all_pairs = find_lss_outputs(args.data_dir)
    all_events = find_lss_events(args.data_dir)
    if not all_pairs:
        if all_events:
            print(f"\n[错误] 找到 {len(all_events)} 个事件文件，但没有对应的 beta NIfTI 文件。")
            print("需要先运行 run_lss_glm.py 生成 beta 文件。")
        else:
            print(f"\n[错误] 数据目录中没有找到 LSS 输出: {args.data_dir}")
        sys.exit(1)

    # 发现被试
    subjects = sorted({p.parts[-3] for p in all_events})
    print(f"发现 {len(subjects)} 个被试: {', '.join(subjects)}")
    print(f"找到 {len(all_pairs)} 个 beta 文件")

    # 单被试测试
    single_results = []
    for subject in subjects:
        result = train_single_subject(
            args.data_dir, subject,
            feature_k=args.feature_k,
            max_voxels=args.max_voxels,
            svm_c=args.svm_c,
        )
        single_results.append(result)

    # 跨被试测试
    cross_result = train_cross_subject(
        args.data_dir,
        feature_k=args.feature_k,
        max_voxels=args.max_voxels,
        svm_c=args.svm_c,
    )

    # 保存最终模型
    save_final_model(
        args.data_dir, args.model_dir,
        feature_k=args.feature_k,
        max_voxels=args.max_voxels,
        svm_c=args.svm_c,
    )

    # 生成结果图表
    plot_results(single_results, cross_result, args.output)

    # 保存结果
    results = {
        "single_subject": convert_numpy(single_results),
        "cross_subject": convert_numpy(cross_result),
        "config": {
            "data_dir": str(args.data_dir),
            "feature_k": args.feature_k,
            "max_voxels": args.max_voxels,
            "svm_c": args.svm_c,
        },
    }

    results_path = args.output / "test_results.json"
    args.output.mkdir(parents=True, exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"测试结果已保存到: {results_path}")

    # 打印汇总
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"{'被试':<10} {'Top-1':<10} {'Top-3':<10} {'Top-5':<10}")
    print("-" * 40)
    for r in single_results:
        if r.get("status") == "ok":
            print(f"{'S' + r['subject']:<10} {r['top1_acc']:<10.2%} "
                  f"{r['top3_acc']:<10.2%} {r['top5_acc']:<10.2%}")
        else:
            print(f"{'S' + r['subject']:<10} {'跳过':<10}")
    if cross_result.get("status") == "ok":
        print(f"{'跨被试':<10} {cross_result['top1_acc']:<10.2%} "
              f"{cross_result['top3_acc']:<10.2%} {cross_result['top5_acc']:<10.2%}")
    print(f"{'Chance':<10} {1/13:<10.2%} {3/13:<10.2%} {5/13:<10.2%}")
    print("=" * 70)


if __name__ == "__main__":
    main()
