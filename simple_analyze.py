#!/usr/bin/env python3
"""
Ultra-simple trajectory analysis script for Recovery-Bench
Features:
1. Direct OpenAI API integration (no terminal-bench dependencies)
2. Regex pre-check to minimize LLM calls (efficiency/cost)
3. Concurrent processing with ThreadPoolExecutor
4. Targets specific error types: NameError, TypeError, AttributeError, application failures
5. Creates recovery traces with clean commands before first target error
"""

import os
import json
import logging
import re
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import load_dotenv
import openai

# Use absolute paths for robustness
BASE_DIR = Path(__file__).parent.resolve()
TARGET_TASK = "4ab82a39-chess-best-move"
BASE_RUNS_FOLDER = BASE_DIR / "runs/gpt-4o-mini-collected-20250714_232243"
SAVE_POINTS_FOLDER = BASE_DIR / "save_points"

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


def llm_check_error(output: str) -> bool:
    """Use LLM to detect if output contains one of the target error types."""
    
    # 1. Increase token limit for reliability (was 5, now 10)
    prompt = f"""Does this terminal output contain one of these specific error types?

1. Undefined names/symbols in runtime (NameError, undefined variable, "is not defined")
2. Application-level failure reported (fatal error, application error, segmentation fault, core dumped)
3. Type/attribute errors at runtime (TypeError, AttributeError, "has no attribute", "unsupported operand type")

Output: {output}

Answer ONLY "Yes" or "No"."""

    try:
        # Small delay to avoid rate limiting
        time.sleep(0.1)
        
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )
        response_text = response.choices[0].message.content
        
        return "yes" in response_text.strip().lower()
    except Exception as e:
        logging.error(f"LLM API call failed: {e}")
        return False

def is_target_error(output: str) -> bool:
    """
    Combined check: uses regex for a fast filter, then uses LLM if necessary.
    """
    if not output.strip():
        return False
    
    # Fast Pre-Check: If regex matches, use LLM to confirm error type
    if ERROR_REGEX.search(output):
        return llm_check_error(output)
        
    # Skip very short outputs (likely just prompts)
    if len(output.split('\n')) < 3 and not output.strip().startswith(('Traceback', 'Error')):
        return False

    # Final check for ambiguous outputs
    return llm_check_error(output)

# analyze trajectory

def analyze_episode(episode_path: Path):
    """
    Analyze one episode and return clean commands before first target error.
    Returns a tuple: (episode_path, clean_commands)
    """
    debug_file = episode_path / "debug.json"
    if not debug_file.exists():
        return episode_path, []
    
    try:
        with open(debug_file, 'r') as f:
            debug_data = json.load(f)
    except Exception as e:
        logging.error(f"Failed to read/load JSON from {debug_file}: {e}")
        return episode_path, []
    
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
        if is_target_error(step["output"]):
            logging.info(f"Error found in {episode_path.name}")
            return episode_path, clean_commands
        else:
            clean_commands.extend(step["commands"])
            
    # if the episode completed without a detected target error
    return episode_path, clean_commands

def save_recovery_trace(episode_path: Path, clean_commands: list):
    """Save the recovery trace to the structured save points folder."""
    # Ensure save paths are relative to BASE_RUNS_FOLDER for a clean structure
    relative_path = episode_path.relative_to(BASE_RUNS_FOLDER)
    save_path = SAVE_POINTS_FOLDER / relative_path
    save_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        "original_episode": str(episode_path),
        "clean_commands": clean_commands,
        "total_clean_commands": len(clean_commands)
    }
    with open(save_path / "recovery_trace.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved {len(clean_commands)} commands for {episode_path.name}")

def main():
    """Main function: finds trajectories and processes episodes concurrently."""
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
    task_path = BASE_RUNS_FOLDER / TARGET_TASK
    if not task_path.exists():
        logging.error(f"Task not found: {task_path.name}")
        return
    logging.info(f"Processing task: {TARGET_TASK}")
    trajectories = [item for item in task_path.iterdir() 
                    if item.is_dir() and (item / "agent-logs").is_dir()]
    if not trajectories:
        logging.error(f"No trajectories found in {task_path.name}")
        return
    traj_path = trajectories[0]
    agent_logs = traj_path / "agent-logs"
    episodes = sorted(
        [item for item in agent_logs.iterdir() if item.is_dir() and item.name.startswith("episode-")],
        key=lambda x: int(x.name.split("-")[1])
    )
    if not episodes:
        logging.error(f"No episodes found in {agent_logs.name}")
        return
    logging.info(f"Found {len(episodes)} episodes")
    
    max_workers = 2 
    save_points_created = 0

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_episode = {
            executor.submit(analyze_episode, episode_path): episode_path
            for episode_path in episodes
        }
        
        for future in as_completed(future_to_episode):
            episode_path = future_to_episode[future]
            
            try:
                _, clean_commands = future.result()
            except Exception as e:
                logging.error(f"Error processing {episode_path.name}: {e}")
                continue

            if clean_commands:
                save_recovery_trace(episode_path, clean_commands)
                save_points_created += 1
    
    logging.info(f"Complete: {save_points_created}/{len(episodes)} save points created")

if __name__ == "__main__":
    main()