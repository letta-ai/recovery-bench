import json
import re
from pathlib import Path
import argparse


def parse_episodes_replayed(file_path):
    """Parse episodes_replayed.txt to extract the number of replayed episodes."""
    try:
        with open(file_path, 'r') as f:
            content = f.read().strip()
        # Extract number using regex from format "replayed X episodes from..."
        match = re.search(r'replayed (\d+) episodes', content)
        if match:
            return int(match.group(1))
        return 0
    except FileNotFoundError:
        return 0


def count_episodes_in_directory(agent_logs_dir):
    """Count episode directories in agent-logs."""
    if not agent_logs_dir.exists():
        return 0
    
    episode_count = 0
    for item in agent_logs_dir.iterdir():
        if item.is_dir() and item.name.startswith("episode-"):
            episode_count += 1
    return episode_count


def extract_task_ids_from_experiment(experiment_folder):
    """Extract task_ids from experiment folder directory names."""
    task_ids = []
    for item in experiment_folder.iterdir():
        if item.is_dir() and '-' in item.name:
            # Extract task_id from "hash-task_name" format
            task_id = '-'.join(item.name.split('-')[1:])
            task_ids.append(task_id)
    return sorted(task_ids)


def analyze_experiment(experiment_folder):
    """Analyze experiment results and generate statistics."""
    experiment_base = Path(experiment_folder)
    
    # Read results.json to get resolved status
    results_file = experiment_base / "results.json"
    if not results_file.exists():
        raise FileNotFoundError(f"results.json not found in {experiment_base}")
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    resolved_ids = set(results.get("resolved_ids", []))
    
    # Extract task_ids from directory structure
    task_ids = extract_task_ids_from_experiment(experiment_base)
    
    experiment_results = []
    
    for task_id in task_ids:
        # Find task directory (hash-task_name format)
        task_dir = None
        for item in experiment_base.iterdir():
            if item.is_dir() and item.name.endswith(f"-{task_id}"):
                task_dir = item
                break
        
        if not task_dir:
            continue
            
        # Find the execution directory (task_name.1-of-1.experiment_name)
        execution_dir = None
        for item in task_dir.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_id}.1-of-1"):
                execution_dir = item
                break
        
        if not execution_dir:
            continue
            
        agent_logs_dir = execution_dir / "agent-logs"
        
        # Count total episodes
        total_episodes = count_episodes_in_directory(agent_logs_dir)
        
        # Count replayed episodes
        episodes_replayed_file = agent_logs_dir / "episodes_replayed.txt"
        replayed_episodes = parse_episodes_replayed(episodes_replayed_file)
        
        # Calculate experiment-generated episodes
        experiment_episodes = total_episodes - replayed_episodes
        
        # Get resolved status
        resolved = "resolved" if task_id in resolved_ids else "unresolved"
        
        experiment_results.append([
            task_id,
            replayed_episodes,
            experiment_episodes,
            total_episodes,
            resolved
        ])
    
    return experiment_results


def main():
    parser = argparse.ArgumentParser(description="Analyze self-correction experiment results")
    parser.add_argument("--experiment_folder", type=str, required=True,
                        help="Path to the experiment folder")
    args = parser.parse_args()
    
    try:
        experiment_results = analyze_experiment(args.experiment_folder)
        
        # Sort by resolved status (resolved first), then by total episodes
        experiment_results.sort(key=lambda x: (x[4] != "resolved", x[3]))
        
        # Print results table
        print("Task ID".ljust(50), "Replayed".ljust(12), "Generated".ljust(12), 
              "Total".ljust(12), "Status".ljust(12))
        print("-" * 98)
        
        for result in experiment_results:
            task_id, replayed, generated, total, status = result
            print(f"{task_id:<50} {replayed:<12} {generated:<12} {total:<12} {status:<12}")
        
        # Print summary
        resolved_count = sum(1 for r in experiment_results if r[4] == "resolved")
        total_count = len(experiment_results)
        print(f"\nSummary: {resolved_count}/{total_count} tasks resolved ({resolved_count/total_count*100:.1f}%)")
        
    except Exception as e:
        print(f"Error analyzing experiment: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())


