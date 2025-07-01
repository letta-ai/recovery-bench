#!/usr/bin/env python3
"""
Reorganize benchmark results by task description hash.

Usage: python hash_reorganize.py <model_path> [--execute]
Example: python hash_reorganize.py runs/sample-claude-3-5-haiku

Note: Run this script in the 'tbench' conda environment:
conda activate tbench
"""

import os
import sys
import hashlib
import shutil
from pathlib import Path
from typing import Optional, Dict

from terminal_bench.handlers.trial_handler import Task


def extract_task_description(task_name: str) -> Optional[str]:
    """Extract task description from terminal-bench task.yaml file."""
    try:
        task_path = Path('/home/kevinlin/terminal-bench/tasks') / task_name / 'task.yaml'
        if not task_path.exists():
            print(f"Warning: Task file not found for {task_name}")
            return None
        
        task = Task.from_yaml(task_path)
        return task.instruction
    
    except Exception as e:
        print(f"Error reading task {task_name}: {e}")
        return None


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode('utf-8')).hexdigest()[:8]


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


def reorganize_directories(base_path: str, dry_run: bool = True) -> None:
    """Reorganize directories by task hash."""
    print(f"{'[DRY RUN] ' if dry_run else ''}Reorganizing {base_path}")
    
    task_dirs = find_task_directories(base_path)
    if not task_dirs:
        print("No task directories found")
        return
    
    task_to_hash = {}
    
    # Process task descriptions
    for task_dir, task_name in task_dirs.items():
        task_description = extract_task_description(task_name)
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
        hash_dir = base_dir / task_hash
        new_task_dir = hash_dir / task_name
        
        if not dry_run:
            try:
                hash_dir.mkdir(exist_ok=True)
                shutil.move(str(task_dir), str(new_task_dir))
                print(f"    Moved to {hash_dir.name}/{task_name}")
            except Exception as e:
                print(f"    Error moving {task_dir}: {e}")
    
    print(f"Processed {len(task_to_hash)} tasks")


def main():
    """Main function."""
    if len(sys.argv) < 2:
        print("Usage: python hash_reorganize.py <model_path> [--execute]")
        print("Add --execute to perform reorganization (default is dry run)")
        print("Note: Run in 'tbench' conda environment")
        sys.exit(1)
    
    base_path = sys.argv[1]
    dry_run = '--execute' not in sys.argv
    
    if dry_run:
        print("DRY RUN MODE: Use --execute to make changes")
    
    reorganize_directories(base_path, dry_run=dry_run)


if __name__ == "__main__":
    main()
