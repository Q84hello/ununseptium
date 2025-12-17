#!/usr/bin/env python3
"""Generate deterministic figures for ununseptium documentation.

This script generates figures for documentation using fixed random seeds
to ensure reproducibility. All figures are saved to docs/figures/.

Figures generated:
1. calibration_curve.svg - Model calibration plot
2. roc_pr_curves.svg - ROC and Precision-Recall curves
3. drift_over_time.svg - Data/model drift visualization
4. evt_tail_plot.svg - Extreme Value Theory tail plot
5. latency_histogram.svg - Latency distribution
6. motif_frequency.svg - Graph motif frequencies
7. throughput_benchmark.svg - Throughput/latency benchmarks
8. audit_chain.svg - Audit log hash chain illustration

Usage:
    python scripts/generate_figures.py [--output-dir PATH] [--format FORMAT]

Requirements:
    - matplotlib
    - numpy
    - scipy
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Check for required dependencies
try:
    import matplotlib
    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy import stats
except ImportError as e:
    print(f"Missing required dependency: {e}")
    print("Install with: pip install matplotlib numpy scipy")
    sys.exit(1)

# Global random seed for reproducibility
RANDOM_SEED = 42


def set_style() -> None:
    """Set consistent matplotlib style for all figures."""
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.size": 10,
        "axes.titlesize": 12,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 150,
        "savefig.dpi": 150,
        "savefig.bbox": "tight",
    })


def generate_calibration_curve(output_dir: Path, fmt: str) -> None:
    """Generate model calibration curve."""
    np.random.seed(RANDOM_SEED)

    # Simulated calibration data
    n_bins = 10
    mean_predicted = np.linspace(0.05, 0.95, n_bins)

    # Well-calibrated model
    fraction_positives_good = mean_predicted + np.random.normal(0, 0.02, n_bins)
    fraction_positives_good = np.clip(fraction_positives_good, 0, 1)

    # Poorly calibrated model (overconfident)
    fraction_positives_bad = mean_predicted ** 2 + np.random.normal(0, 0.03, n_bins)
    fraction_positives_bad = np.clip(fraction_positives_bad, 0, 1)

    fig, ax = plt.subplots(figsize=(6, 5))

    # Perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect Calibration")

    # Model curves
    ax.plot(mean_predicted, fraction_positives_good, "o-", color="#2E86AB",
            label="Model A (Well Calibrated)")
    ax.plot(mean_predicted, fraction_positives_bad, "s-", color="#E94F37",
            label="Model B (Overconfident)")

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve Comparison")
    ax.legend(loc="lower right")
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    output_path = output_dir / f"calibration_curve.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_roc_pr_curves(output_dir: Path, fmt: str) -> None:
    """Generate ROC and Precision-Recall curves."""
    np.random.seed(RANDOM_SEED + 1)

    # Simulated model scores
    n_samples = 1000
    y_true = np.random.binomial(1, 0.3, n_samples)

    # Good model
    scores_good = y_true * 0.7 + np.random.normal(0.5, 0.15, n_samples)
    scores_good = np.clip(scores_good, 0, 1)

    # Mediocre model
    scores_med = y_true * 0.4 + np.random.normal(0.5, 0.2, n_samples)
    scores_med = np.clip(scores_med, 0, 1)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # ROC Curve
    for scores, label, color in [
        (scores_good, "Model A (AUC=0.92)", "#2E86AB"),
        (scores_med, "Model B (AUC=0.75)", "#E94F37"),
    ]:
        thresholds = np.linspace(0, 1, 100)
        tpr_list = []
        fpr_list = []

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))
            tn = np.sum((preds == 0) & (y_true == 0))

            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
            tpr_list.append(tpr)
            fpr_list.append(fpr)

        axes[0].plot(fpr_list, tpr_list, label=label, color=color, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend(loc="lower right")

    # Precision-Recall Curve
    for scores, label, color in [
        (scores_good, "Model A (AP=0.88)", "#2E86AB"),
        (scores_med, "Model B (AP=0.65)", "#E94F37"),
    ]:
        thresholds = np.linspace(0, 1, 100)
        precision_list = []
        recall_list = []

        for thresh in thresholds:
            preds = (scores >= thresh).astype(int)
            tp = np.sum((preds == 1) & (y_true == 1))
            fp = np.sum((preds == 1) & (y_true == 0))
            fn = np.sum((preds == 0) & (y_true == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 1
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision_list.append(precision)
            recall_list.append(recall)

        axes[1].plot(recall_list, precision_list, label=label, color=color, linewidth=2)

    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision-Recall Curve")
    axes[1].legend(loc="lower left")

    plt.tight_layout()
    output_path = output_dir / f"roc_pr_curves.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_drift_over_time(output_dir: Path, fmt: str) -> None:
    """Generate data/model drift visualization."""
    np.random.seed(RANDOM_SEED + 2)

    # Time periods (months)
    months = np.arange(1, 13)
    month_labels = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    # Simulated drift metrics
    feature_drift = 0.1 + 0.05 * np.log(months) + np.random.normal(0, 0.02, 12)
    performance_auc = 0.92 - 0.02 * np.log(months) + np.random.normal(0, 0.01, 12)
    prediction_drift = 0.05 + 0.03 * months / 12 + np.random.normal(0, 0.01, 12)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # Feature drift
    axes[0].bar(months, feature_drift, color="#2E86AB", alpha=0.8)
    axes[0].axhline(y=0.2, color="#E94F37", linestyle="--", label="Threshold")
    axes[0].set_xlabel("Month")
    axes[0].set_ylabel("PSI Score")
    axes[0].set_title("Feature Distribution Drift")
    axes[0].set_xticks(months)
    axes[0].set_xticklabels(month_labels, rotation=45)
    axes[0].legend()

    # Performance drift
    axes[1].plot(months, performance_auc, "o-", color="#2E86AB", linewidth=2)
    axes[1].fill_between(months, performance_auc - 0.02, performance_auc + 0.02,
                          alpha=0.2, color="#2E86AB")
    axes[1].axhline(y=0.85, color="#E94F37", linestyle="--", label="Minimum AUC")
    axes[1].set_xlabel("Month")
    axes[1].set_ylabel("AUC-ROC")
    axes[1].set_title("Model Performance Over Time")
    axes[1].set_xticks(months)
    axes[1].set_xticklabels(month_labels, rotation=45)
    axes[1].legend()

    # Prediction drift
    axes[2].plot(months, prediction_drift, "s-", color="#2E86AB", linewidth=2)
    axes[2].axhline(y=0.1, color="#E94F37", linestyle="--", label="Alert Threshold")
    axes[2].set_xlabel("Month")
    axes[2].set_ylabel("KL Divergence")
    axes[2].set_title("Prediction Distribution Drift")
    axes[2].set_xticks(months)
    axes[2].set_xticklabels(month_labels, rotation=45)
    axes[2].legend()

    plt.tight_layout()
    output_path = output_dir / f"drift_over_time.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_evt_tail_plot(output_dir: Path, fmt: str) -> None:
    """Generate Extreme Value Theory tail plot."""
    np.random.seed(RANDOM_SEED + 3)

    # Generate data from Generalized Pareto Distribution
    xi = 0.2  # Shape parameter
    sigma = 1.0  # Scale parameter
    n = 1000

    # GPD samples
    uniform = np.random.uniform(0, 1, n)
    gpd_samples = sigma / xi * ((1 - uniform) ** (-xi) - 1)

    # Sort for tail plot
    gpd_sorted = np.sort(gpd_samples)
    exceedance_prob = 1 - np.arange(1, n + 1) / (n + 1)

    # Theoretical tail
    x_theory = np.linspace(0.1, gpd_sorted[-1], 200)
    tail_theory = (1 + xi * x_theory / sigma) ** (-1 / xi)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Empirical vs Theoretical tail
    axes[0].semilogy(gpd_sorted, exceedance_prob, ".", alpha=0.5, label="Empirical", color="#2E86AB")
    axes[0].semilogy(x_theory, tail_theory, "-", linewidth=2, label="GPD Fit", color="#E94F37")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("P(X > x)")
    axes[0].set_title("Tail Probability Plot")
    axes[0].legend()
    axes[0].set_ylim([1e-4, 1])

    # QQ Plot
    theoretical_quantiles = sigma / xi * ((1 - np.linspace(0.001, 0.999, n)) ** (-xi) - 1)
    theoretical_quantiles = np.sort(theoretical_quantiles)

    axes[1].plot(theoretical_quantiles, gpd_sorted, ".", alpha=0.5, color="#2E86AB")
    max_val = max(theoretical_quantiles.max(), gpd_sorted.max())
    axes[1].plot([0, max_val], [0, max_val], "k--", label="Perfect Fit")
    axes[1].set_xlabel("Theoretical Quantiles")
    axes[1].set_ylabel("Empirical Quantiles")
    axes[1].set_title("QQ Plot (GPD)")
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / f"evt_tail_plot.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_latency_histogram(output_dir: Path, fmt: str) -> None:
    """Generate latency distribution histogram."""
    np.random.seed(RANDOM_SEED + 4)

    # Simulated latency data (log-normal distribution)
    latencies_fast = np.random.lognormal(mean=2, sigma=0.5, size=5000)
    latencies_slow = np.random.lognormal(mean=3, sigma=0.7, size=5000)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    # Histogram
    axes[0].hist(latencies_fast, bins=50, alpha=0.7, label="Optimized", color="#2E86AB", density=True)
    axes[0].hist(latencies_slow, bins=50, alpha=0.7, label="Baseline", color="#E94F37", density=True)
    axes[0].axvline(x=np.percentile(latencies_fast, 99), color="#2E86AB", linestyle="--",
                    label=f"P99 Opt: {np.percentile(latencies_fast, 99):.1f}ms")
    axes[0].axvline(x=np.percentile(latencies_slow, 99), color="#E94F37", linestyle="--",
                    label=f"P99 Base: {np.percentile(latencies_slow, 99):.1f}ms")
    axes[0].set_xlabel("Latency (ms)")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Latency Distribution")
    axes[0].legend()
    axes[0].set_xlim([0, 150])

    # Percentile comparison
    percentiles = [50, 75, 90, 95, 99]
    fast_pcts = [np.percentile(latencies_fast, p) for p in percentiles]
    slow_pcts = [np.percentile(latencies_slow, p) for p in percentiles]

    x = np.arange(len(percentiles))
    width = 0.35

    axes[1].bar(x - width / 2, fast_pcts, width, label="Optimized", color="#2E86AB")
    axes[1].bar(x + width / 2, slow_pcts, width, label="Baseline", color="#E94F37")
    axes[1].set_xlabel("Percentile")
    axes[1].set_ylabel("Latency (ms)")
    axes[1].set_title("Latency Percentiles")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([f"P{p}" for p in percentiles])
    axes[1].legend()

    plt.tight_layout()
    output_path = output_dir / f"latency_histogram.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_motif_frequency(output_dir: Path, fmt: str) -> None:
    """Generate graph motif frequency plot."""
    np.random.seed(RANDOM_SEED + 5)

    # Motif types and counts
    motifs = ["Triangle", "Star-3", "Path-3", "Cycle-4", "Clique-4", "Star-4"]
    normal_counts = [150, 280, 420, 85, 25, 120]
    suspicious_counts = [380, 150, 280, 220, 95, 180]

    # Add some noise
    normal_counts = [c + np.random.randint(-10, 10) for c in normal_counts]
    suspicious_counts = [c + np.random.randint(-10, 10) for c in suspicious_counts]

    fig, ax = plt.subplots(figsize=(8, 5))

    x = np.arange(len(motifs))
    width = 0.35

    bars1 = ax.bar(x - width / 2, normal_counts, width, label="Normal Accounts",
                   color="#2E86AB", alpha=0.8)
    bars2 = ax.bar(x + width / 2, suspicious_counts, width, label="Suspicious Accounts",
                   color="#E94F37", alpha=0.8)

    ax.set_xlabel("Motif Type")
    ax.set_ylabel("Frequency")
    ax.set_title("Graph Motif Frequency Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(motifs, rotation=15)
    ax.legend()

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{int(height)}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    output_path = output_dir / f"motif_frequency.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_throughput_benchmark(output_dir: Path, fmt: str) -> None:
    """Generate throughput/latency benchmark plot."""
    np.random.seed(RANDOM_SEED + 6)

    # Batch sizes and corresponding metrics
    batch_sizes = [1, 8, 16, 32, 64, 128, 256, 512]

    # Throughput (items/sec)
    throughput = [100, 650, 1100, 1800, 2800, 3900, 4800, 5200]
    throughput = [t + np.random.randint(-50, 50) for t in throughput]

    # Latency (ms)
    latency = [10, 12, 14, 18, 24, 35, 55, 100]
    latency = [l + np.random.uniform(-1, 1) for l in latency]

    fig, ax1 = plt.subplots(figsize=(8, 5))

    # Throughput on left axis
    color1 = "#2E86AB"
    ax1.set_xlabel("Batch Size")
    ax1.set_ylabel("Throughput (items/sec)", color=color1)
    line1 = ax1.plot(batch_sizes, throughput, "o-", color=color1, linewidth=2,
                     label="Throughput")
    ax1.tick_params(axis="y", labelcolor=color1)
    ax1.set_xscale("log", base=2)

    # Latency on right axis
    ax2 = ax1.twinx()
    color2 = "#E94F37"
    ax2.set_ylabel("P95 Latency (ms)", color=color2)
    line2 = ax2.plot(batch_sizes, latency, "s--", color=color2, linewidth=2,
                     label="P95 Latency")
    ax2.tick_params(axis="y", labelcolor=color2)

    # Combined legend
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc="upper left")

    ax1.set_title("Throughput vs Latency Trade-off")
    ax1.set_xticks(batch_sizes)
    ax1.set_xticklabels(batch_sizes)

    plt.tight_layout()
    output_path = output_dir / f"throughput_benchmark.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def generate_audit_chain(output_dir: Path, fmt: str) -> None:
    """Generate audit log hash chain illustration."""
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 4])
    ax.axis("off")

    # Draw blocks
    block_positions = [(1, 2), (4, 2), (7, 2)]
    block_labels = ["Entry 1", "Entry 2", "Entry 3"]
    hashes = ["H1 = SHA256(...)", "H2 = SHA256(H1||E2)", "H3 = SHA256(H2||E3)"]

    for i, ((x, y), label, hash_text) in enumerate(zip(block_positions, block_labels, hashes)):
        # Block rectangle
        rect = plt.Rectangle((x - 0.8, y - 0.5), 1.6, 1.5, fill=True,
                              facecolor="#2E86AB", edgecolor="#1A5276", linewidth=2)
        ax.add_patch(rect)

        # Block label
        ax.text(x, y + 0.3, label, ha="center", va="center", fontsize=10,
                fontweight="bold", color="white")

        # Hash text below
        ax.text(x, y - 0.15, hash_text, ha="center", va="center", fontsize=7,
                color="white", style="italic")

        # Arrow to next block
        if i < len(block_positions) - 1:
            ax.annotate("", xy=(block_positions[i + 1][0] - 0.9, y),
                        xytext=(x + 0.9, y),
                        arrowprops=dict(arrowstyle="->", color="#E94F37", lw=2))

    # Title
    ax.text(5, 3.5, "Tamper-Evident Audit Log (Hash Chain)", ha="center", va="center",
            fontsize=12, fontweight="bold")

    # Description
    ax.text(5, 0.5, "Each entry hash includes the previous hash, creating an immutable chain",
            ha="center", va="center", fontsize=9, style="italic", color="#555555")

    plt.tight_layout()
    output_path = output_dir / f"audit_chain.{fmt}"
    fig.savefig(output_path)
    plt.close(fig)
    print(f"Generated: {output_path}")


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Generate documentation figures")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent.parent / "docs" / "figures",
        help="Output directory for figures",
    )
    parser.add_argument(
        "--format",
        choices=["svg", "png"],
        default="svg",
        help="Output format (default: svg)",
    )

    args = parser.parse_args()

    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Set matplotlib style
    set_style()

    print(f"Generating figures in {args.output_dir} (format: {args.format})\n")

    # Generate all figures
    generate_calibration_curve(args.output_dir, args.format)
    generate_roc_pr_curves(args.output_dir, args.format)
    generate_drift_over_time(args.output_dir, args.format)
    generate_evt_tail_plot(args.output_dir, args.format)
    generate_latency_histogram(args.output_dir, args.format)
    generate_motif_frequency(args.output_dir, args.format)
    generate_throughput_benchmark(args.output_dir, args.format)
    generate_audit_chain(args.output_dir, args.format)

    print(f"\nGenerated 8 figures successfully!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
