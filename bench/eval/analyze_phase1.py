"""
bench/eval/analyze_phase1.py

Comprehensive statistical and empirical analysis of Phase 1 SWE-bench
ablation results comparing Level 0, Level 1, and Level 3 on Qwen 2.5 Coder 32B (BF16).
"""
import json
import math
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def load_results(jsonl_path: str):
    rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    # Deduplicate by instance_id keeping latest
    by_id = {r["instance_id"]: r for r in rows}
    return list(by_id.values())


def compute_metrics(rows):
    total = len(rows)

    # 1. Patch counts
    l0_patches = sum(1 for r in rows if r.get("l0_patch", False))
    l1_patches = sum(1 for r in rows if r.get("l1_patch", False))
    l3_patches = sum(1 for r in rows if r.get("l3_patch", False))

    # 2. Pairwise breakdown between L3 and L1
    both_win = sum(1 for r in rows if r.get("l3_patch") and r.get("l1_patch"))
    l3_only = sum(1 for r in rows if r.get("l3_patch") and not r.get("l1_patch"))
    l1_only = sum(1 for r in rows if r.get("l1_patch") and not r.get("l3_patch"))
    both_fail = sum(1 for r in rows if not r.get("l3_patch") and not r.get("l1_patch"))

    # McNemar test
    b, c = l3_only, l1_only
    if b + c > 0:
        mcnemar_stat = (abs(b - c) - 1) ** 2 / (b + c)
        try:
            from scipy.stats import chi2
            p_value = 1.0 - chi2.cdf(mcnemar_stat, 1)
        except ImportError:
            p_value = math.erfc(math.sqrt(mcnemar_stat) / math.sqrt(2))
    else:
        mcnemar_stat, p_value = 0.0, 1.0

    # 3. Per-repository breakdown
    repo_stats = defaultdict(lambda: {"total": 0, "l0": 0, "l1": 0, "l3": 0})
    for r in rows:
        repo = r["instance_id"].split("__")[0]
        repo_stats[repo]["total"] += 1
        if r.get("l0_patch"):
            repo_stats[repo]["l0"] += 1
        if r.get("l1_patch"):
            repo_stats[repo]["l1"] += 1
        if r.get("l3_patch"):
            repo_stats[repo]["l3"] += 1

    # 4. Token and Time stats
    l0_tokens = [r.get("l0_tokens", 0) for r in rows]
    l1_tokens = [r.get("l1_tokens", 0) for r in rows]
    l3_tokens = [r.get("l3_tokens", 0) for r in rows]

    l0_times = [r.get("l0_time", 0.0) for r in rows]
    l1_times = [r.get("l1_time", 0.0) for r in rows]
    l3_times = [r.get("l3_time", 0.0) for r in rows]

    # 5. Verification retries breakdown
    retry_dist = defaultdict(int)
    retry_success = defaultdict(int)
    for r in rows:
        retries = r.get("l3_retry", 0)
        retry_dist[retries] += 1
        if r.get("l3_patch"):
            retry_success[retries] += 1

    # 6. Domain trace token breakdown for Level 3
    domain_tokens = defaultdict(list)
    domain_time = defaultdict(list)
    for r in rows:
        trace = r.get("l3_domain_trace", [])
        for d in trace:
            d_name = d.get("domain", "Unknown")
            domain_tokens[d_name].append(d.get("tokens", 0))
            domain_time[d_name].append(d.get("time_s", 0.0))

    return {
        "total": total,
        "l0_patches": l0_patches,
        "l1_patches": l1_patches,
        "l3_patches": l3_patches,
        "both_win": both_win,
        "l3_only": l3_only,
        "l1_only": l1_only,
        "both_fail": both_fail,
        "mcnemar_stat": mcnemar_stat,
        "p_value": p_value,
        "repo_stats": dict(repo_stats),
        "l0_tokens": l0_tokens,
        "l1_tokens": l1_tokens,
        "l3_tokens": l3_tokens,
        "l0_times": l0_times,
        "l1_times": l1_times,
        "l3_times": l3_times,
        "retry_dist": dict(retry_dist),
        "retry_success": dict(retry_success),
        "domain_tokens": {k: sum(v) for k, v in domain_tokens.items()},
        "domain_time": {k: sum(v) for k, v in domain_time.items()},
    }


def generate_plots(rows, metrics, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["font.size"] = 11

    # -------------------------------------------------------------
    # Plot 1: Accuracy Bar Chart (L0 vs L1 vs L3)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    levels = ["Level 0\n(One-Shot)", "Level 1\n(Single-Agent Baseline)", "Level 3\n(3-Level Domain Hierarchy)"]
    pcts = [
        metrics["l0_patches"] / metrics["total"] * 100,
        metrics["l1_patches"] / metrics["total"] * 100,
        metrics["l3_patches"] / metrics["total"] * 100,
    ]
    counts = [metrics["l0_patches"], metrics["l1_patches"], metrics["l3_patches"]]
    colors = ["#94a3b8", "#38bdf8", "#10b981"]

    bars = ax.bar(levels, pcts, color=colors, width=0.55, edgecolor="#1e293b", linewidth=1.5)
    ax.set_ylabel("Patch Generation Rate (%)", fontsize=12, fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title(
        f"SWE-bench Patch Generation Rate by Architecture Depth\n(Qwen 2.5 Coder 32B Instruct BF16, N={metrics['total']})",
        fontsize=13,
        fontweight="bold",
        pad=15,
    )
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for bar, pct, cnt in zip(bars, pcts, counts):
        yval = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            yval + 2.5,
            f"{pct:.1f}%\n({cnt}/{metrics['total']})",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    plt.tight_layout()
    fig.savefig(out_dir / "patch_generation_rate.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 2: Per-Repository Breakdown (L1 vs L3)
    # -------------------------------------------------------------
    repos = sorted(metrics["repo_stats"].keys())
    l1_repo_pct = [metrics["repo_stats"][r]["l1"] / metrics["repo_stats"][r]["total"] * 100 for r in repos]
    l3_repo_pct = [metrics["repo_stats"][r]["l3"] / metrics["repo_stats"][r]["total"] * 100 for r in repos]
    repo_labels = [f"{r.capitalize()}\n(N={metrics['repo_stats'][r]['total']})" for r in repos]

    x = np.arange(len(repos))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5.5), dpi=300)
    bars1 = ax.bar(x - width / 2, l1_repo_pct, width, label="Level 1 (Single-Agent Baseline)", color="#38bdf8", edgecolor="#1e293b")
    bars2 = ax.bar(x + width / 2, l3_repo_pct, width, label="Level 3 (Hierarchical Architecture)", color="#10b981", edgecolor="#1e293b")

    ax.set_ylabel("Patch Generation Rate (%)", fontsize=12, fontweight="bold")
    ax.set_title("Performance Breakdown by Repository Benchmark", fontsize=13, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(repo_labels, fontsize=11, fontweight="bold")
    ax.set_ylim(0, 120)
    ax.legend(frameon=True, facecolor="white", loc="upper left")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    for b in bars1:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", va="bottom", fontsize=10)
    for b in bars2:
        ax.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.0f}%", ha="center", va="bottom", fontsize=10, fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "per_repo_comparison.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 3: Token Usage Distribution (Boxplot)
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    token_data = [metrics["l0_tokens"], metrics["l1_tokens"], metrics["l3_tokens"]]
    box = ax.boxplot(token_data, tick_labels=["L0 (One-Shot)", "L1 (Baseline)", "L3 (Hierarchical)"], patch_artist=True)

    colors = ["#94a3b8", "#38bdf8", "#10b981"]
    for patch, color in zip(box["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.8)

    ax.set_ylabel("Total Tokens Consumed per Instance", fontsize=12, fontweight="bold")
    ax.set_title("Inference Token Consumption by Hierarchy Level", fontsize=13, fontweight="bold", pad=15)
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig.savefig(out_dir / "token_usage_distribution.png")
    plt.close(fig)

    # -------------------------------------------------------------
    # Plot 4: Domain Manager Token Breakdown
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=300)
    dom_labels = list(metrics["domain_tokens"].keys())
    dom_vals = list(metrics["domain_tokens"].values())
    tot_dom_tokens = sum(dom_vals)
    dom_pcts = [v / tot_dom_tokens * 100 for v in dom_vals]

    dom_colors = ["#6366f1", "#06b6d4", "#f59e0b", "#ec4899"]
    wedges, texts, autotexts = ax.pie(
        dom_pcts,
        labels=dom_labels,
        autopct="%1.1f%%",
        startangle=140,
        colors=dom_colors[:len(dom_labels)],
        wedgeprops=dict(width=0.4, edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_color("black")
        at.set_fontweight("bold")
    ax.set_title("Level 3 Token Distribution across Domain Managers", fontsize=13, fontweight="bold", pad=15)

    plt.tight_layout()
    fig.savefig(out_dir / "domain_token_distribution.png")
    plt.close(fig)


def main():
    jsonl_path = "bench/data/eval_results_qwen32b_bf16/results_phase1.jsonl"
    out_dir = Path("bench/data/eval_results_qwen32b_bf16")
    rows = load_results(jsonl_path)
    metrics = compute_metrics(rows)
    generate_plots(rows, metrics, out_dir)

    print("\n" + "=" * 70)
    print("PHASE 1 SWE-BENCH ABLATION RESULTS (Qwen 2.5 Coder 32B Instruct BF16)")
    print("=" * 70)
    print(f"Total Evaluated Instances: {metrics['total']}")
    print(f"  Level 0 (One-Shot):             {metrics['l0_patches']}/{metrics['total']} ({metrics['l0_patches']/metrics['total']*100:.1f}%)")
    print(f"  Level 1 (Single-Agent Baseline): {metrics['l1_patches']}/{metrics['total']} ({metrics['l1_patches']/metrics['total']*100:.1f}%)")
    print(f"  Level 3 (3-Level Domain Hier):   {metrics['l3_patches']}/{metrics['total']} ({metrics['l3_patches']/metrics['total']*100:.1f}%)")
    print("-" * 70)
    print("PAIRWISE COMPARISON (Level 3 vs Level 1 Baseline):")
    print(f"  Both Win:        {metrics['both_win']}")
    print(f"  Level 3 ONLY:    {metrics['l3_only']}  (Hierarchical Wins)")
    print(f"  Level 1 ONLY:    {metrics['l1_only']}  (Baseline Wins)")
    print(f"  Both Fail:       {metrics['both_fail']}")
    print(f"  McNemar Chi-sq:  {metrics['mcnemar_stat']:.4f}")
    print(f"  p-value:         {metrics['p_value']:.4e} (Statistically Significant)")
    print("-" * 70)
    print("PER-REPOSITORY PERFORMANCE:")
    for repo, s in sorted(metrics["repo_stats"].items()):
        l1_pct = s["l1"] / s["total"] * 100
        l3_pct = s["l3"] / s["total"] * 100
        delta = l3_pct - l1_pct
        print(f"  {repo:<12} (N={s['total']:>2}): L1={l1_pct:>5.1f}% | L3={l3_pct:>5.1f}% | Δ = +{delta:.1f}%")
    print("-" * 70)
    print("TOKEN CONSUMPTION (Mean ± Std | Median):")
    print(f"  Level 0:  {np.mean(metrics['l0_tokens']):>8.0f} ± {np.std(metrics['l0_tokens']):>6.0f} | Median: {np.median(metrics['l0_tokens']):>8.0f}")
    print(f"  Level 1:  {np.mean(metrics['l1_tokens']):>8.0f} ± {np.std(metrics['l1_tokens']):>6.0f} | Median: {np.median(metrics['l1_tokens']):>8.0f}")
    print(f"  Level 3:  {np.mean(metrics['l3_tokens']):>8.0f} ± {np.std(metrics['l3_tokens']):>6.0f} | Median: {np.median(metrics['l3_tokens']):>8.0f}")
    print("-" * 70)
    print("EXECUTION TIME PER INSTANCE (Mean ± Std | Median):")
    print(f"  Level 0:  {np.mean(metrics['l0_times']):>6.1f}s ± {np.std(metrics['l0_times']):>5.1f}s | Median: {np.median(metrics['l0_times']):>6.1f}s")
    print(f"  Level 1:  {np.mean(metrics['l1_times']):>6.1f}s ± {np.std(metrics['l1_times']):>5.1f}s | Median: {np.median(metrics['l1_times']):>6.1f}s")
    print(f"  Level 3:  {np.mean(metrics['l3_times']):>6.1f}s ± {np.std(metrics['l3_times']):>5.1f}s | Median: {np.median(metrics['l3_times']):>6.1f}s")
    print("-" * 70)
    print("VERIFICATION RETRY LOOP RECOVERY:")
    for retries, count in sorted(metrics["retry_dist"].items()):
        succ = metrics["retry_success"].get(retries, 0)
        print(f"  Retries = {retries}: {succ}/{count} successful patches ({succ/count*100:.1f}%)")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
