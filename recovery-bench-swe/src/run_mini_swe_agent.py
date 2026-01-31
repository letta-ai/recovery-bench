"""
Run replay/recovery agent.

This script runs the replay agent on collected trajectories for swebench.
"""

import argparse
import sys
from pathlib import Path
from .swe_utils import (run_replay_agent_swe, run_default_agent_swe)

def main():
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent for all trajectories"
    )

    #Dirty State Trajectories for weak mini-swe-agent
    parser.add_argument(
        "--trajectory-folder",
        type=str,
        required=True,
        help="Path to the trajectory folder",
    )

    #Valid mini-swe-agent model
    parser.add_argument(
        "--model-name", 
        type=str, 
        required=True, 
        help="Name of the model to use"
    )

    #RunID if user specifies
    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help='ID associated with a run'
    )

    #Specify the max number of workers
    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help='Max Workers for SWEBench Evaluator'
    )

    #Specify how much history context wanted for mini-swe-agent
    parser.add_argument(
        "--recovery-mode",
        choices=["environment_only", "action_summary", "full_history"],
        default="full_history",
        help="How much context to provide to recovery agent"
    )

    parser.add_argument(
        "--default-agent",
        action="store_true",
        help="Run's the Default Agent on the Recovery-Bench dataset"
    )

    #Check if the path to mini-swe-agent trajectories is valid
    args, unknown_args = parser.parse_known_args()
    trajectory_folder = Path(args.trajectory_folder)
    if not trajectory_folder or not trajectory_folder.exists():
        print(f"Error: Mini-SWE-Agent trajectory folder {args.trajectory_folder} does not exist")
        return 1
    
    #Make CWD recovery-swe-agent
    cwd = Path(__file__).parent.parent
    
    #Collect args for run_replay_agent_swe
    model = args.model_name
    recovery_mode = args.recovery_mode
    run_id = args.run_id
    max_workers = args.max_workers
    default_agent = args.default_agent
    if default_agent:
        return run_default_agent_swe(cwd, trajectory_folder, model, run_id, max_workers)
    return run_replay_agent_swe(cwd, trajectory_folder, model, run_id, recovery_mode, max_workers)
    
if __name__ == "__main__":
    sys.exit(main())
