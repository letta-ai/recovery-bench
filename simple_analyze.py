#!/usr/bin/env python3
"""
Ultra-simple trajectory analysis script for Recovery-Bench
Features:
1. Direct OpenAI API integration (no terminal-bench dependencies)
2. Regex pre-check to minimize LLM calls (efficiency/cost)
3. Concurrent processing with ThreadPoolExecutor
4. Targets specific error types: NameError, TypeError, AttributeError, application failures
5. Creates recovery traces with clean commands before first target error
6. EXCLUDES: "command not found" and "timeout" errors (these do NOT count as errors)
7. Outputs error types for each command that has errors (excluding the main three)
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

def save_recovery_trace(episode_path: Path, clean_commands: list, error_type: str):
    """Save the recovery trace to the structured save points folder."""
    # Ensure save paths are relative to BASE_RUNS_FOLDER for a clean structure
    relative_path = episode_path.relative_to(BASE_RUNS_FOLDER)
    save_path = SAVE_POINTS_FOLDER / relative_path
    save_path.mkdir(parents=True, exist_ok=True)
    
    data = {
        "original_episode": str(episode_path),
        "clean_commands": clean_commands,
        "total_clean_commands": len(clean_commands),
        "error_type": error_type
    }
    with open(save_path / "recovery_trace.json", 'w') as f:
        json.dump(data, f, indent=2)
    
    logging.info(f"Saved {len(clean_commands)} commands for {episode_path.name} with error type: {error_type}")

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
    
    save_points_created = 0

    # Sequentially process episodes; stop at the first episode containing an error
    for episode_path in episodes:
        try:
            _, clean_commands, error_type = analyze_episode(episode_path)
        except Exception as e:
            logging.error(f"Error processing {episode_path.name}: {e}")
            continue

        if error_type:
            # Always save a recovery trace for the first error episode,
            # even if there are zero clean commands before the error
            save_recovery_trace(episode_path, clean_commands, error_type)
            save_points_created += 1
            print(f"Stopping at episode {episode_path.name} due to error: {error_type}")
            break
    
    logging.info(f"Complete: {save_points_created}/{len(episodes)} save points created")

if __name__ == "__main__":
    main()