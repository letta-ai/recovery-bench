"""
This script is for getting descriptive statistics regarding the 
dirty state trajectories
"""

import argparse
import json
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from pathlib import Path
from swe_utils import get_data
from .generate_traces import (get_runs_dir, get_swe_root_dir, cat_dir_and_model)
#from recovery_bench_swe.src.generate_traces import (get_runs_dir, get_swe_root_dir, cat_dir_and_model)
from swebench.harness.run_evaluation import main as run_swe_bench

JSON = "json"
JSONL = "jsonl"
TXT = "txt"
DATASET = "princeton-nlp/SWE-bench_Verified"
PREDICTIONS_PATH = Path("predictions")
EVAL_RESULTS_PATH = Path("evaluation-results")
END_CMD = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT && git add -A && git diff --cached"

def get_dirty_state_instances(model: str):
    instances = []
    swe_runs = cat_dir_and_model(get_runs_dir(), model)
    print(f"SWE RUNS: {swe_runs}")
    for p in swe_runs.iterdir():
        commands_path = p / "commands.json"
        try:
            with open(str(commands_path), "r") as f:
                data = json.load(f)
                instances.append(data["instance"])
        except Exception as e:
            print(f"Failed to open {commands_path}")
    return instances


def check_predictions(swe_dir):
    predictions_path = swe_dir / "predictions/baseline.jsonl"
    data = []
    try:
        with open(str(predictions_path), 'r') as f:
            for line in f:
                data.append(json.loads(line))
        instances = []
        for traj in data:
            instances.append(traj["instance_id"])
        
        return run_swe_bench(
            dataset_name = DATASET,
            split = "test",
            instance_ids=instances,
            predictions_path = str(predictions_path),
            max_workers = 8,
            force_rebuild = False,
            cache_level = "env",
            clean = True,
            open_file_limit = 4096,
            run_id = "baseline",
            timeout = 1800,
            namespace = "swebench",
            rewrite_reports = False,
            modal = False,
            instance_image_tag = "latest",
            env_image_tag = "latest",
        )
        
    except Exception as e:
        print("failed to read ", predictions_path)
        print(f"Error: {e}")


#Since recovery data is smaller as of now, hard code knowing that information
def match_instances(base_data, rec_data):
    for inst in base_data["submitted_ids"]:
        # if inst == "django__django-13568":
        #     continue

        if inst not in rec_data["submitted_ids"]:
            base_data["submitted_ids"].remove(inst)
            if inst in base_data["resolved_ids"]:
                base_data["resolved_ids"].remove(inst)
            elif inst in base_data["unresolved_ids"]:
                base_data["unresolved_ids"].remove(inst)
            elif inst in base_data["error_ids"].remove(inst):
                base_data["error_ids"].remove(inst)
            elif inst in base_data["empty_patch_ids"]:
                base_data["empty_patch_ids"].remove(inst)    

def plot_total_commands(default_total: int, recovery_total: int):
    """Plot total commands comparison."""
    diff = recovery_total - default_total
    pct_change = (diff / default_total * 100) if default_total > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    labels = ["Clean (baseline)", "Dirty (recovery)"]
    values = [default_total, recovery_total]
    colors = ['#6B7FD7', '#C86A5A']
    
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#333333', alpha=0.5)
    ax.set_axisbelow(True)
    
    ax.set_ylabel("Total Commands", fontsize=12, color='white')
    ax.tick_params(colors='white', labelsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 10,
            f'{val}',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold',
            color='white'
        )
    
    # Add difference annotation
    rec_bar = bars[1]
    ax.text(
        rec_bar.get_x() + rec_bar.get_width() / 2,
        recovery_total / 2,
        f'+{diff}',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color='#E85D4D'
    )

    title_text = "Total Commands: Clean vs Dirty State"
    subtitle_text = f"Takeaway: +{diff} commands (+{pct_change:.1f}% increase) in dirty state"
    
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color='white',
        y=0.98,
        x=0.5
    )
    
    ax.text(
        0.5, 1.08,
        subtitle_text,
        transform=ax.transAxes,
        fontsize=11,
        color='#AAAAAA',
        ha='center',
        va='top'
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    out_path = get_swe_root_dir() / "total_commands_comparison.png"
    plt.savefig(out_path, dpi=200, facecolor='#1a1a1a', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved total commands chart to {out_path}")


def plot_first_tokens(default_counter: Counter, recovery_counter: Counter, top_n: int = 15):
    """Plot first tokens comparison with grouped bars."""
    # Get top tokens by total usage
    all_tokens = set(default_counter.keys()) | set(recovery_counter.keys())
    token_totals = {token: default_counter[token] + recovery_counter[token] for token in all_tokens}
    top_tokens = sorted(token_totals.keys(), key=lambda t: token_totals[t], reverse=True)[:top_n]
    
    # Prepare data
    default_counts = [default_counter[token] for token in top_tokens]
    recovery_counts = [recovery_counter[token] for token in top_tokens]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    x = np.arange(len(top_tokens))
    width = 0.35
    
    colors = ['#6B7FD7', '#C86A5A']
    
    bars1 = ax.bar(x - width/2, default_counts, width, label='Clean (baseline)', 
                   color=colors[0], edgecolor='none')
    bars2 = ax.bar(x + width/2, recovery_counts, width, label='Dirty (recovery)', 
                   color=colors[1], edgecolor='none')

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#333333', alpha=0.5)
    ax.set_axisbelow(True)
    
    ax.set_xlabel("Command Token", fontsize=12, color='white')
    ax.set_ylabel("Frequency", fontsize=12, color='white')
    ax.set_xticks(x)
    ax.set_xticklabels(top_tokens, rotation=45, ha='right')
    ax.tick_params(colors='white', labelsize=10)
    
    # Add legend
    ax.legend(facecolor='#2a2a2a', edgecolor='#666666', 
              fontsize=10, labelcolor='white', loc='upper right')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add label if there's a value
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    height + 2,
                    f'{int(height)}',
                    ha='center',
                    va='bottom',
                    fontsize=8,
                    color='white'
                )
    
    add_labels(bars1)
    add_labels(bars2)

    title_text = f"Top {top_n} Command Tokens: Clean vs Dirty State"
    
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color='white',
        y=0.96,
        x=0.5
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.94])
    
    out_path = get_swe_root_dir() / "first_tokens_comparison.png"
    plt.savefig(out_path, dpi=200, facecolor='#1a1a1a', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved first tokens chart to {out_path}")


def plot_api_calls(default_avg: float, recovery_avg: float):
    """Plot API calls comparison."""
    diff = recovery_avg - default_avg
    pct_change = (diff / default_avg * 100) if default_avg > 0 else 0
    
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    labels = ["Clean (baseline)", "Dirty (recovery)"]
    values = [default_avg, recovery_avg]
    colors = ['#6B7FD7', '#C86A5A']
    
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#333333', alpha=0.5)
    ax.set_axisbelow(True)
    
    ax.set_ylabel("Average API Calls", fontsize=12, color='white')
    ax.tick_params(colors='white', labelsize=11)
    
    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 0.3,
            f'{val:.2f}',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold',
            color='white'
        )
    
    # Add difference annotation
    rec_bar = bars[1]
    ax.text(
        rec_bar.get_x() + rec_bar.get_width() / 2,
        recovery_avg / 2,
        f'+{diff:.2f}',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color='#E85D4D'
    )

    title_text = "API Calls: Clean vs Dirty State"
    subtitle_text = f"Takeaway: +{diff:.2f} calls (+{pct_change:.1f}% increase) in dirty state"
    
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color='white',
        y=0.98,
        x=0.5
    )
    
    ax.text(
        0.5, 1.08,
        subtitle_text,
        transform=ax.transAxes,
        fontsize=11,
        color='#AAAAAA',
        ha='center',
        va='top'
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    out_path = get_swe_root_dir() / "api_calls_comparison.png"
    plt.savefig(out_path, dpi=200, facecolor='#1a1a1a', edgecolor='none')
    plt.close(fig)
    
    print(f"Saved API calls chart to {out_path}")

def compare_api_call_stats(default_runs: Path, recovery_runs: Path, common_submitted: set):
    """Compare API call statistics between default and recovery agents."""
    default_api_calls = []
    recovery_api_calls = []
    
    for instance_id in common_submitted:
        # Load default trajectory
        default_traj_path = default_runs / instance_id / f"{instance_id}.traj.json"
        if default_traj_path.exists():
            with open(default_traj_path) as f:
                default_traj = json.load(f)
                default_api_calls.append(default_traj.get("info", {}).get("model_stats", {}).get("api_calls", 0))
        
        # Load recovery trajectory
        recovery_traj_path = recovery_runs / instance_id / f"{instance_id}.traj.json"
        if recovery_traj_path.exists():
            with open(recovery_traj_path) as f:
                recovery_traj = json.load(f)
                recovery_api_calls.append(recovery_traj.get("info", {}).get("model_stats", {}).get("api_calls", 0))
    
    # Calculate averages
    default_avg = sum(default_api_calls) / len(default_api_calls) if default_api_calls else 0
    recovery_avg = sum(recovery_api_calls) / len(recovery_api_calls) if recovery_api_calls else 0
    
    print(f"\n=== API Call Statistics ===")
    print(f"Default agent average API calls: {default_avg:.2f}")
    print(f"Recovery agent average API calls: {recovery_avg:.2f}")
    print(f"Difference: {recovery_avg - default_avg:+.2f} ({(recovery_avg - default_avg) / default_avg * 100:+.2f}%)")

    plot_api_calls(default_avg, recovery_avg)

    return default_avg, recovery_avg


def compare_first_tokens(default_runs: Path, recovery_runs: Path, common_submitted: set):
    """Compare first tokens of commands between default and recovery agents."""
    
    default_first_tokens = []
    recovery_first_tokens = []
    
    for instance_id in common_submitted:
        # Load default commands
        default_commands_path = default_runs / instance_id / "commands.json"
        if default_commands_path.exists():
            with open(default_commands_path) as f:
                commands = json.load(f)
                for cmd in commands:
                    # Get first token (word) from each command
                    first_token = cmd.split()[0] if cmd.strip() else ""
                    if first_token:
                        default_first_tokens.append(first_token)
        
        # Load recovery commands
        recovery_commands_path = recovery_runs / instance_id / "commands.json"
        if recovery_commands_path.exists():
            with open(recovery_commands_path) as f:
                commands = json.load(f)
                for cmd in commands:
                    first_token = cmd.split()[0] if cmd.strip() else ""
                    if first_token:
                        recovery_first_tokens.append(first_token)
    
    # Count frequencies
    default_counter = Counter(default_first_tokens)
    recovery_counter = Counter(recovery_first_tokens)
    
    # Get all unique first tokens
    all_tokens = set(default_counter.keys()) | set(recovery_counter.keys())
    
    print(f"\n=== First Token Comparison ===")
    print(f"{'Token':<15} {'Default':<10} {'Recovery':<10} {'Difference':<12} {'% Change':<12}")
    print("-" * 65)
    
    # Sort by most common in recovery
    sorted_tokens = sorted(all_tokens, key=lambda t: recovery_counter[t], reverse=True)
    
    for token in sorted_tokens[:20]: 
        default_count = default_counter[token]
        recovery_count = recovery_counter[token]
        diff = recovery_count - default_count
        pct_change = (diff / default_count * 100) if default_count > 0 else float('inf')
        
        print(f"{token:<15} {default_count:<10} {recovery_count:<10} {diff:+<12} {pct_change:+.1f}%")
    
    # Summary statistics
    print(f"\n=== Summary ===")
    print(f"Total commands (default): {len(default_first_tokens)}")
    print(f"Total commands (recovery): {len(recovery_first_tokens)}")
    print(f"Unique tokens (default): {len(default_counter)}")
    print(f"Unique tokens (recovery): {len(recovery_counter)}")

    plot_first_tokens(default_counter, recovery_counter, top_n=15)
    plot_total_commands(len(default_first_tokens), len(recovery_first_tokens))


def plot(baseline_pct, recovery_pct):
    drop_abs = baseline_pct - recovery_pct
    drop_rel = drop_abs / baseline_pct * 100 if baseline_pct > 0 else 0.0

    # Create figure with dark background like the reference
    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor('#1a1a1a')
    ax.set_facecolor('#1a1a1a')

    labels = ["Clean (baseline)", "Dirty (recovery)"]
    values = [baseline_pct, recovery_pct]
    
    # Colors matching the reference style
    colors = ['#6B7FD7', '#C86A5A']
    
    # Create bars
    bars = ax.bar(labels, values, color=colors, width=0.5, edgecolor='none')

    # Set y-axis limits: start at 0, cap at 60
    ax.set_ylim(0, 60)

    # Style the axes
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#666666')
    ax.spines['bottom'].set_color('#666666')
    
    # Grid styling
    ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='#333333', alpha=0.5)
    ax.set_axisbelow(True)
    
    # Labels and title styling
    ax.set_ylabel("Success rate (%)", fontsize=12, color='white')
    ax.tick_params(colors='white', labelsize=11)
    
    # Add percentage labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + 1.5,
            f'{val:.1f}%',
            ha='center',
            va='bottom',
            fontsize=13,
            fontweight='bold',
            color='white'
        )
    
    # Add drop percentage annotation above recovery bar (like the red numbers in reference)
    rec_bar = bars[1]
    ax.text(
        rec_bar.get_x() + rec_bar.get_width() / 2,
        recovery_pct / 2,
        f'−{drop_abs:.1f}%',
        ha='center',
        va='center',
        fontsize=12,
        fontweight='bold',
        color='#E85D4D'
    )

    # Title styling
    title_text = "Recovery-Bench: Clean vs Dirty State Success Rate"
    subtitle_text = f"Takeaway: −{drop_abs:.1f} pts (−{drop_rel:.1f}% relative) performance drop in dirty state"
    
    fig.suptitle(
        title_text,
        fontsize=16,
        fontweight='bold',
        color='white',
        y=0.98,
        x=0.5
    )
    
    ax.text(
        0.5, 1.08,
        subtitle_text,
        transform=ax.transAxes,
        fontsize=11,
        color='#AAAAAA',
        ha='center',
        va='top'
    )

    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    out_path = get_swe_root_dir() / "recovery_vs_baseline(1).png"
    plt.savefig(out_path, dpi=200, facecolor='#1a1a1a', edgecolor='none')
    plt.close(fig)

    print(f"\nSaved comparison chart to {out_path}")


def recovery_vs_baseline(default_results_path: Path, 
                         recovery_results_path: Path,
                         default_runs: Path,
                         recovery_runs: Path):

    default_data, recovery_data = {}, {}
    try:
        with open(default_results_path) as f:
            default_data = json.load(f)
    
        with open(recovery_results_path) as f:
            recovery_data = json.load(f)
        
        # Find common submitted instances
        default_submitted = set(default_data['submitted_ids'])
        recovery_submitted = set(recovery_data['submitted_ids'])
        recovery_errors = set(recovery_data.get('error_ids', []))
        common_submitted = (default_submitted & recovery_submitted) - recovery_errors

        default_resolved = set(default_data['resolved_ids']) & common_submitted
        recovery_resolved = set(recovery_data['resolved_ids']) & common_submitted

        n_common = len(common_submitted)
        default_rate = len(default_resolved) / n_common if n_common > 0 else 0
        recovery_rate = len(recovery_resolved) / n_common if n_common > 0 else 0
        
        print(f"Common submitted instances: {n_common}")
        print(f"Default resolved: {len(default_resolved)}/{n_common} ({default_rate:.2%})")
        print(f"Recovery resolved: {len(recovery_resolved)}/{n_common} ({recovery_rate:.2%})")
        print(f"Difference: {recovery_rate - default_rate:+.2%}")
    

        default_pct = default_rate * 100
        recovery_pct = recovery_rate * 100
        plot(default_pct, recovery_pct)
        compare_api_call_stats(default_runs, recovery_runs, common_submitted)
        compare_first_tokens(default_runs, recovery_runs, common_submitted)

        return 0


    except Exception as e:
        print(f"ERROR, failed to open recovery or baseline data")
        return 1
    
def find_failed_patches(logs_path: Path):
    success_patches = 0
    for p in logs_path.iterdir():
        try:
            report_path = p / "report.json"
            report = get_data(report_path, JSON)
            for key in report.keys():
                if report[key]["patch_successfully_applied"]:
                    success_patches += 1
                else:
                    print(f"{report_path} failed to apply patch\n")
        except Exception as e:
            print("no report, ", p)
    print(f"Number of successful patches returned: {success_patches}")

def get_matching_instances(default_path: Path, recovery_path: Path):
    recovery_files = {f.name for f in recovery_path.iterdir() if f.is_file()}
    default_files = {f.name for f in default_path.iterdir() if f.is_file()}
    
    # Get intersection of both sets
    matching_files = recovery_files & default_files
    
    return matching_files

def main():
    parser = argparse.ArgumentParser(
        description="Generate statistics and plot differences between Default and Recovery Agent"
    )

    parser.add_argument(
        "--recovery-trajs",
        type=str,
        help="Path to desired recovery agent trajectories run"
    )

    parser.add_argument(
        "--default-trajs",
        type=str,
        help="Path to desired default agent trajectories run"
    )

    parser.add_argument(
        "--recovery-results",
        type=str,
        help="Path to desired recovery agent results json"
    )

    parser.add_argument(
        "--default-results",
        type=str,
        help="Path to desired default agent results json"
    )

    args, unknown_args = parser.parse_known_args()


    recovery_trajs = Path(args.recovery_trajs)
    default_trajs = Path(args.default_trajs)
    recovery_results = Path(args.recovery_results)
    default_results = Path(args.default_results)

    recovery_vs_baseline(default_results, 
                         recovery_results, 
                         default_trajs, 
                         recovery_trajs)

if __name__ == "__main__":
    main()
