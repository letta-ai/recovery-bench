"""
Run replay/recovery agent or default agent on collected trajectories for SWE-Bench.
"""

import argparse
import sys
from pathlib import Path
from .swe_utils import run_swe_agent

def main():
    parser = argparse.ArgumentParser(
        description="Run mini-swe-agent for all trajectories"
    )

    parser.add_argument(
        "--trajectory-folder",
        type=str,
        required=True,
        help="Path to the trajectory folder",
    )

    parser.add_argument(
        "--model", 
        type=str, 
        required=True, 
        help="Name of the minisweagent litellmm compatible model to use"
    )

    parser.add_argument(
        "--run-id",
        type=str,
        required=True,
        help='Unique ID associated with a run'
    )

    parser.add_argument(
        "--max-workers",
        type=int,
        default=1,
        help='Max workers for SWEBench Evaluator'
    )

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

    args, unknown_args = parser.parse_known_args()
    trajectory_folder = Path(args.trajectory_folder)
    if not trajectory_folder or not trajectory_folder.exists():
        print(f"Error: Mini-SWE-Agent trajectory folder {args.trajectory_folder} does not exist")
        return 1
    
    #Make CWD recovery-bench-swe
    swe_root = Path(__file__).parent.parent
    model = args.model
    recovery_mode = args.recovery_mode
    run_id = args.run_id
    max_workers = args.max_workers
    default_agent = args.default_agent
    return run_swe_agent(cwd=swe_root, 
                        trajectory_folder=trajectory_folder, 
                        model=model, 
                        run_id=run_id, 
                        recovery_mode=recovery_mode, 
                        max_workers=max_workers,
                        default_agent=default_agent)
    
if __name__ == "__main__":
    sys.exit(main())