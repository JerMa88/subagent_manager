import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import argparse
from pathlib import Path

LEVEL_NAMES = {
    0: "L0 (One-Shot)",
    1: "L1 (Baseline)",
    2: "L2 (Flat Multi-Agent)",
    3: "L3 (Hierarchical)",
    4: "L4 (Deep Phase)",
}

def plot_ablation(jsonl_path, output_dir="bench/results"):
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    data = []
    raw_rows = []
    with open(jsonl_path, "r") as f:
        for line in f:
            if line.strip():
                try:
                    raw_rows.append(json.loads(line))
                except Exception:
                    pass

    latest_by_id = {}
    for r in raw_rows:
        latest_by_id[r["instance_id"]] = r
    unique_rows = list(latest_by_id.values())

    for row in unique_rows:
        inst_id = row.get("instance_id", "")
        for lvl in range(5):
            prefix = f"l{lvl}_"
            if f"{prefix}patch" not in row:
                continue

            patch = row[f"{prefix}patch"]
            time_s = row.get(f"{prefix}time", 0.0)
            tokens = row.get(f"{prefix}tokens", 0)
            subtasks = max(row.get(f"{prefix}subtasks", 0), 1)

            speed = tokens / time_s if time_s > 0 else 0.0
            avg_tokens = tokens / subtasks if subtasks > 0 else tokens

            data.append({
                "Instance": inst_id,
                "Level": LEVEL_NAMES[lvl],
                "Level_ID": lvl,
                "Accuracy": 1.0 if patch else 0.0,
                "Time (s)": time_s,
                "Speed (tokens/s)": speed,
                "Total Tokens": tokens,
                "Avg Tokens per Agent": avg_tokens,
                "Effective Depth": row.get(f"l{lvl}_depth", lvl),
            })

    if not data:
        print("No ablation data parsed!")
        return

    df = pd.DataFrame(data)

    sns.set_theme(style="whitegrid", palette="muted")

    # 1. Accuracy Bar Plot across all 5 Levels
    plt.figure(figsize=(10, 6))
    acc_df = df.groupby("Level")["Accuracy"].mean().reset_index()
    # Sort by Level_ID order
    acc_df["Level_ID"] = acc_df["Level"].map({v: k for k, v in LEVEL_NAMES.items()})
    acc_df = acc_df.sort_values("Level_ID")

    ax = sns.barplot(data=acc_df, x="Level", y="Accuracy", hue="Level", palette="viridis", legend=False)
    plt.title("SWE-bench Accuracy by Hierarchy Depth Level (L0-L4)", fontsize=14, fontweight="bold")
    plt.ylabel("Accuracy (Patch Generation %)", fontsize=12)
    plt.xlabel("Hierarchy Depth Level", fontsize=12)
    plt.ylim(0, 1.0)

    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{height*100:.1f}%", (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=11, fontweight="bold", xytext=(0, 3), textcoords="offset points")

    plt.tight_layout()
    plt.savefig(out_path / "ablation_accuracy_bar.png", dpi=300)
    plt.close()

    # 2. Box & Violin Plots for Metrics
    metrics = [
        ("Time (s)", "Total Time to Finish (s)", "ablation_time"),
        ("Speed (tokens/s)", "Token Generation Speed (tokens/s)", "ablation_speed"),
        ("Total Tokens", "Total Tokens Generated", "ablation_tokens"),
        ("Avg Tokens per Agent", "Average Tokens per Agent", "ablation_avg_tokens"),
    ]

    for col, title, file_prefix in metrics:
        # Box Plot
        plt.figure(figsize=(11, 6))
        sns.boxplot(data=df, x="Level", y=col, hue="Level", palette="viridis", legend=False)
        plt.title(f"Box Plot: {title}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path / f"{file_prefix}_box.png", dpi=300)
        plt.close()

        # Violin Plot
        plt.figure(figsize=(11, 6))
        sns.violinplot(data=df, x="Level", y=col, hue="Level", palette="viridis", inner="quartile", legend=False)
        plt.title(f"Violin Plot: {title}", fontsize=14, fontweight="bold")
        plt.tight_layout()
        plt.savefig(out_path / f"{file_prefix}_violin.png", dpi=300)
        plt.close()

    summary_df = df.groupby("Level").agg(
        Accuracy=("Accuracy", lambda x: f"{x.mean()*100:.1f}%"),
        Avg_Time_s=("Time (s)", lambda x: f"{x.mean():.1f}s"),
        Avg_Tokens=("Total Tokens", lambda x: f"{x.mean():.0f}"),
        Effective_Depth=("Effective Depth", lambda x: f"{x.mean():.2f}")
    ).reset_index()
    print("\n" + "="*70)
    print(summary_df.to_string(index=False))
    print("="*70 + "\n")

    print(f"Ablation plots successfully updated and saved in {output_dir}/!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="bench/results/ablation_300.jsonl")
    parser.add_argument("--output-dir", type=str, default="bench/results")
    args = parser.parse_args()
    plot_ablation(args.input, args.output_dir)
