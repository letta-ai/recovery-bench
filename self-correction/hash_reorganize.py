#!/usr/bin/env python3
"""
Reorganize benchmark results by task description hash.

Usage: python hash_reorganize.py <model_path> [--reverse]
Example: python hash_reorganize.py runs/sample-claude-3-5-haiku

Note: Run this script in the 'tbench' conda environment:
conda activate tbench
"""

import shutil
import argparse
from pathlib import Path
import os
from typing import Optional, Dict

from .utils import create_task_hash
from terminal_bench.handlers.trial_handler import Task


def extract_task_description(task_name: str, task_folder: str = None) -> Optional[str]:
    """Extract task description from terminal-bench task.yaml file."""
    try:
        # Use task_folder argument first, then fall back to environment variable
        folder = task_folder if task_folder is not None else os.getenv("TASK_FOLDER")
        if folder is None:
            print(f"Warning: No task folder specified for {task_name}")
            return None
        
        task_path = Path(folder) / task_name / 'task.yaml'
        if not task_path.exists():
            print(f"Warning: Task file not found for {task_name}")
            return None
        
        task = Task.from_yaml(task_path)
        return task.instruction
    
    except Exception as e:
        print(f"Error reading task {task_name}: {e}")
        return None


def find_task_directories(base_path: str) -> Dict[str, str]:
    """Find task directories with their names."""
    task_dirs = {}
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return task_dirs
    
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.endswith(('.json', '.log', '.lock')):
            # Check if this looks like a task directory (has the expected subdirectory structure)
            task_subdir = None
            for subitem in item.iterdir():
                if subitem.is_dir() and subitem.name.startswith(item.name + '.'):
                    task_subdir = subitem
                    break
            
            # If we found the expected structure, use the task name
            if task_subdir:
                task_dirs[str(item)] = item.name
            else:
                # Handle special case where directory structure might be different
                # Still include it as a potential task directory
                task_dirs[str(item)] = item.name
    
    return task_dirs


def find_hash_directories(base_path: str) -> Dict[str, list]:
    """Find directories with hash prefixes."""
    hash_dirs = {}
    base_path = Path(base_path)
    
    if not base_path.exists():
        print(f"Error: Path {base_path} does not exist")
        return hash_dirs
    
    for item in base_path.iterdir():
        if item.is_dir() and not item.name.startswith('.') and not item.name.endswith(('.json', '.log', '.lock')):
            # Check if this looks like a hash-prefixed directory (8 hex chars + dash + name)
            if len(item.name) > 9 and item.name[8] == '-':
                hash_part = item.name[:8]
                if all(c in '0123456789abcdef' for c in hash_part.lower()):
                    hash_dirs[hash_part] = [str(item)]
    
    return hash_dirs


def is_hash_prefixed_directory(task_dir: str, task_name: str) -> bool:
    """Check if directory is already hash-prefixed by inspecting subdirectory structure."""
    # Check if name looks like hash prefix (8 hex chars + dash)
    if len(task_name) > 9 and task_name[8] == '-':
        hash_part = task_name[:8]
        if all(c in '0123456789abcdef' for c in hash_part.lower()):
            # Extract the part after the dash
            name_after_dash = task_name[9:]
            
            # Look for subdirectory that starts with the name after dash
            task_path = Path(task_dir)
            for subitem in task_path.iterdir():
                if subitem.is_dir() and subitem.name.startswith(name_after_dash + '.'):
                    return True
    
    return False


def reorganize_directories(base_path: str, task_folder: str = None) -> None:
    """Reorganize directories by adding task hash prefix."""
    print(f"Reorganizing {base_path}")
    
    task_dirs = find_task_directories(base_path)
    if not task_dirs:
        print("No task directories found")
        return
    
    task_to_hash = {}
    
    # Process task descriptions
    for task_dir, task_name in task_dirs.items():
        # Check if directory already has hash prefix by inspecting subdirectory structure
        if is_hash_prefixed_directory(task_dir, task_name):
            print(f"  {task_name} -> SKIPPED (already has hash prefix)")
            continue
        
        task_description = extract_task_description(task_name, task_folder)
        if task_description:
            task_hash = create_task_hash(task_description)
            task_to_hash[task_dir] = task_hash
            print(f"  {task_name} -> {task_hash}")
        else:
            print(f"  {task_name} -> SKIPPED (no task definition)")
    
    # Perform reorganization
    for task_dir, task_hash in task_to_hash.items():
        task_name = Path(task_dir).name
        base_dir = Path(task_dir).parent
        new_task_dir = base_dir / f"{task_hash}-{task_name}"
        
        try:
            shutil.move(str(task_dir), str(new_task_dir))
            print(f"    Renamed to {task_hash}-{task_name}")
        except Exception as e:
            print(f"    Error renaming {task_dir}: {e}")
    
    print(f"Processed {len(task_to_hash)} tasks")


def reverse_reorganize_directories(base_path: str) -> None:
    """Reverse reorganization by removing hash prefixes from directory names."""
    print(f"Reversing reorganization in {base_path}")
    
    hash_dirs = find_hash_directories(base_path)
    if not hash_dirs:
        print("No hash-prefixed directories found")
        return
    
    moved_count = 0
    
    # Process hash-prefixed directories
    for hash_prefix, task_dirs in hash_dirs.items():
        for task_dir in task_dirs:
            task_dir_path = Path(task_dir)
            task_name = task_dir_path.name
            
            # Remove hash prefix (8 chars + dash)
            original_name = task_name[9:]  # Remove "12345678-"
            target_dir = task_dir_path.parent / original_name
            
            try:
                shutil.move(str(task_dir), str(target_dir))
                print(f"    Renamed {task_name} back to {original_name}")
                moved_count += 1
            except Exception as e:
                print(f"    Error renaming {task_dir}: {e}")
    
    print(f"Processed {moved_count} tasks")


def main():
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Reorganize benchmark results by task description hash.",
        epilog="Note: Run this script in the 'tbench' conda environment: conda activate tbench"
    )
    
    parser.add_argument(
        'model_path',
        help='Path to the model results directory (e.g., runs/sample-claude-3-5-haiku)'
    )
    
    parser.add_argument(
        '--reverse',
        action='store_true',
        help='Reverse the reorganization by moving task directories back to base level'
    )
    
    parser.add_argument(
        '--task-folder',
        type=str,
        default=None,
        help='Path to the task folder (defaults to TASK_FOLDER environment variable)'
    )
    
    args = parser.parse_args()
    
    base_path = args.model_path
    reverse = args.reverse
    task_folder = args.task_folder
    
    if reverse:
        reverse_reorganize_directories(base_path)
    else:
        reorganize_directories(base_path, task_folder)


if __name__ == "__main__":
    main()
