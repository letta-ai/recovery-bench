#!/usr/bin/env python3
"""
Ultra-simple trajectory analysis script for Recovery-Bench
Features:
1. Direct OpenAI API integration (no terminal-bench dependencies)
2. Regex pre-check to minimize LLM calls (efficiency/cost)
3. Targets specific error types: NameError, TypeError, AttributeError, application failures
4. EXCLUDES: "command not found" and "timeout" errors (these do NOT count as errors)
5. Finds the first episode containing a target error
6. Copies episodes 0 through the errored episode to save_points in ReplayAgent format
7. Outputs error types for the errored episode
"""

import os
import json
import logging
import re
import time
import shutil
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import openai

# Use absolute paths for robustness
BASE_DIR = Path(__file__).parent.resolve()
BASE_RUNS_FOLDER = BASE_DIR / "runs/gpt-4o-mini-collected-20250714_232243"
SAVE_POINTS_FOLDER = BASE_DIR / "save_points"

# Tasks to process (can be modified or set to None to process all)
TARGET_TASKS = [
    "4ab82a39-chess-best-move",
    "1e9d5280-parallelize-graph",
    "0812d36d-npm-conflict-resolution"
]

# Regex for cheap pre-check (case-insensitive)
# Captures common error indicators for the target types
ERROR_REGEX = re.compile(
    r'(name|type|attribute)error|'          # NameError, TypeError, AttributeError
    r'is not defined|'                      # "undefined variable" variant
    r'has no attribute|'                    # "has no attribute" variant
    r'unsupported operand type|'            # "unsupported operand type" variant
    r'fatal error|'                         # Application-level failure
    r'segmentation fault|'
    r'core dumped',
    re.IGNORECASE | re.MULTILINE
)

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Done to be compatible with the ReplayAgent format
def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]


def llm_check_error(output: str) -> tuple[bool, str]:
    """Use LLM to detect if output contains one of the target error types and return error type."""
    
    prompt = f"""Does this terminal output contain an error? If yes, what type of error is it?

IMPORTANT: Do NOT consider these as errors:
- "command not found" errors
- "timeout" errors
- "No module named" errors (these are just missing dependencies)

If there is an error (excluding the above), identify the specific error type.
If no error or only excluded errors, respond with "No error".

Output: {output}

Respond with either:
- "No error" if no errors or only excluded errors
- The specific error type if there is a real error (e.g., "NameError", "TypeError", "AttributeError", "SyntaxError", "RuntimeError", etc.)"""

    try:
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=50,
            temperature=0
        )
        response_text = response.choices[0].message.content.strip()
        
        if "no error" in response_text.lower():
            return False, ""
        else:
            return True, response_text
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return False, ""

def is_target_error(output: str) -> tuple[bool, str]:
    """
    Combined check: uses regex for a fast filter, then uses LLM if necessary.
    Returns (is_error, error_type)
    """
    if not output.strip():
        return False, ""
    
    # Skip very short outputs (likely just prompts)
    if len(output.split('\n')) < 3 and not output.strip().startswith(('Traceback', 'Error')):
        return False, ""

    # Always use LLM to check for errors (excluding command not found and timeout)
    return llm_check_error(output)

# analyze trajectory

def analyze_episode(episode_path: Path):
    """
    Analyze one episode and return clean commands before first target error.
    Returns a tuple: (episode_path, clean_commands, error_type)
    """
    debug_file = episode_path / "debug.json"
    if not debug_file.exists():
        return episode_path, [], ""
    
    try:
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read/load JSON from {debug_file}: {e}")
        return episode_path, [], ""
    
    messages = debug_data.get("input", [])
    
    # data structure to hold the execution steps clearly
    steps = []
    
    # 1. collect all executed commands and associate them with the next terminal output (user message)
    last_commands = []
    
    for msg in messages:
        if msg.get("role") == "assistant":
            try:
                # Catch specific exceptions instead of a bare 'except'
                content_json = json.loads(msg.get("content", ""))
                commands = [cmd.get("keystrokes", "").strip() 
                            for cmd in content_json.get("commands", []) if cmd.get("keystrokes")]
                last_commands.extend(commands)
            except (json.JSONDecodeError, AttributeError, TypeError, KeyError):
                # Ignore non-standard assistant messages (e.g., internal commentary)
                continue
            
        elif msg.get("role") == "user" and last_commands:
            # This 'user' message contains the output of the 'assistant's last commands
            content = msg.get("content", "")
            
            # 2. Extract terminal output for error check
            # Look for the last shell prompt to separate command echo from output
            lines = content.split('\n')
            output_lines = []
            in_output = False
            
            for line in reversed(lines):
                if in_output and line.strip():
                    output_lines.append(line)
                
                # Check for the shell prompt (root@...#) - stop at the last one
                if "root@" in line and "#" in line:
                    in_output = True
                    # Stop after finding the first command echo
                    if len(output_lines) > 0:
                        break
            
            output = '\n'.join(reversed(output_lines)).strip()
            
            steps.append({
                "commands": last_commands,
                "output": output,
                "path": episode_path
            })
            last_commands = []

    clean_commands = []
    for step in steps:
        is_error, error_type = is_target_error(step["output"])
        if is_error:
            logging.info(f"Error found in {episode_path.name}: {error_type}")
            print(f"Episode {episode_path.name}: Error Type - {error_type}")
            return episode_path, clean_commands, error_type
        else:
            clean_commands.extend(step["commands"])
            
    # if the episode completed without a detected target error
    return episode_path, clean_commands, ""

def copy_episodes_for_replay(
    task_hash: str, 
    task_name: str, 
    trajectory_path: Path,
    episodes: list, 
    error_episode_num: int,
    error_type: str
) -> Path:
    """
    Copy episodes 0 through error_episode_num to save_points in ReplayAgent format.
    
    Structure: save_points/{hash}-{task_name}/{task_name}.1-of-1.{run_id}/agent-logs/episode-N/
    """
    # Extract run_id from trajectory path name
    # ex, "chess-best-move.1-of-1.initial-gpt-4o-mini-20250714_232243"
    traj_name = trajectory_path.name
    if ".1-of-1." in traj_name:
        run_id = traj_name.split(".1-of-1.")[-1]
    else:
        # Fallback: use a default run_id
        run_id = "replay-recovery"
    
    # Create save path structure
    base_save_path = SAVE_POINTS_FOLDER / f"{task_hash}-{task_name}"
    traj_save_path = base_save_path / f"{task_name}.1-of-1.{run_id}"
    agent_logs_save = traj_save_path / "agent-logs"
    agent_logs_save.mkdir(parents=True, exist_ok=True)
    
    # Copy episodes 0 through error_episode_num (inclusive)
    episodes_copied = 0
    for i in range(error_episode_num + 1):  # +1 to include errored episode
        if i < len(episodes):
            episode_src = episodes[i]
            episode_dest = agent_logs_save / f"episode-{i}"
            
            # Copy entire episode directory
            if episode_dest.exists():
                shutil.rmtree(episode_dest)
            shutil.copytree(episode_src, episode_dest)
            episodes_copied += 1
            logging.info(f"Copied {episode_src.name} -> {episode_dest.name}")
    
    # Save error metadata for reference
    error_metadata = {
        "error_episode": error_episode_num,
        "error_type": error_type,
        "episodes_copied": episodes_copied,
        "original_trajectory": str(trajectory_path)
    }
    with open(traj_save_path / "error_metadata.json", 'w') as f:
        json.dump(error_metadata, f, indent=2)
    
    logging.info(f"Saved {episodes_copied} episodes (0-{error_episode_num}) for ReplayAgent in {traj_save_path}")
    return traj_save_path

def process_task(task_id: str) -> bool:
    """Process a single task and copy episodes for ReplayAgent. Returns True if error found."""
    task_path = BASE_RUNS_FOLDER / task_id
    if not task_path.exists():
        logging.error(f"Task not found: {task_id}")
        return False
    
    logging.info(f"Processing task: {task_id}")
    trajectories = [item for item in task_path.iterdir() 
                    if item.is_dir() and (item / "agent-logs").is_dir()]
    if not trajectories:
        logging.error(f"No trajectories found in {task_id}")
        return False
    
    traj_path = trajectories[0]
    agent_logs = traj_path / "agent-logs"
    episodes = sorted(
        [item for item in agent_logs.iterdir() if item.is_dir() and item.name.startswith("episode-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    if not episodes:
        logging.error(f"No episodes found in {agent_logs.name}")
        return False
    
    logging.info(f"Found {len(episodes)} episodes for {task_id}")

    # Extract task hash and name from task_id (format: "hash-task-name")
    if "-" in task_id:
        parts = task_id.split("-", 1)  # Split on first "-" only
        task_hash = parts[0]
        task_name = parts[1]
    else:
        # Fallback: use task name as-is
        logging.warning(f"Task ID '{task_id}' doesn't have hash prefix. Using as-is...")
        task_hash = "unknown"
        task_name = task_id

    # Sequentially process episodes; stop at the first episode containing an error
    for idx, episode_path in enumerate(episodes):
        try:
            _, clean_commands, error_type = analyze_episode(episode_path)
        except Exception as e:
            logging.error(f"Error processing {episode_path.name}: {e}")
            continue

        if error_type:
            # Copy episodes 0 through the errored episode for ReplayAgent
            episode_num = idx  # Current episode index
            copy_episodes_for_replay(
                task_hash=task_hash,
                task_name=task_name,
                trajectory_path=traj_path,
                episodes=episodes,
                error_episode_num=episode_num,
                error_type=error_type
            )
            print(f"\nTask {task_id}: Error found in episode {episode_num} ({episode_path.name})")
            print(f"   Error Type: {error_type}")
            print(f"   Copied episodes 0-{episode_num} to save_points for ReplayAgent\n")
            return True
    
    logging.info(f"No errors found in {task_id}")
    return False


def main():
    """Main function: processes multiple tasks."""
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        logging.error("OPENAI_API_KEY not set in environment variables.")
        return
    logging.info("API key loaded")
    SAVE_POINTS_FOLDER.mkdir(exist_ok=True)
    openai.api_key = api_key
    
    try:
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=3,
            temperature=0
        )
        logging.info("LLM connection successful")
    except Exception as e:
        logging.error(f"LLM connection failed: {e}")
        return
    
    if not TARGET_TASKS:
        logging.error("No tasks specified in TARGET_TASKS")
        return
    
    logging.info(f"Processing {len(TARGET_TASKS)} task(s): {TARGET_TASKS}")
    
    save_points_created = 0
    for task_id in TARGET_TASKS:
        try:
            if process_task(task_id):
                save_points_created += 1
        except Exception as e:
            logging.error(f"Failed to process task {task_id}: {e}")
            continue
    
    logging.info(f"\n{'='*60}")
    logging.info(f"Complete: {save_points_created}/{len(TARGET_TASKS)} tasks had errors and were saved")
    logging.info(f"{'='*60}")

if __name__ == "__main__":
    main()