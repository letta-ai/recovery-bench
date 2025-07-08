#!/usr/bin/env python3
"""
Script to compare two result.json files and find IDs where the first one fails 
but the second one succeeds (resolved).
"""

import json
import argparse
import sys

def load_results(filepath):
    """Load results from a JSON file and extract resolved/unresolved IDs."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        resolved_ids = set(data.get('resolved_ids', []))
        unresolved_ids = set(data.get('unresolved_ids', []))
        
        return resolved_ids, unresolved_ids
    except FileNotFoundError:
        print(f"Error: File {filepath} not found")
        sys.exit(1)
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in {filepath}")
        sys.exit(1)
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        sys.exit(1)

def find_improvements(first_resolved, first_unresolved, second_resolved, second_unresolved):
    """
    Find IDs where first result fails but second result succeeds.
    
    Args:
        first_resolved: set of IDs resolved in first file
        first_unresolved: set of IDs unresolved in first file
        second_resolved: set of IDs resolved in second file
        second_unresolved: set of IDs unresolved in second file
        
    Returns:
        set of IDs where first fails and second succeeds
    """
    # IDs that are unresolved in first but resolved in second
    improvements = first_unresolved & second_resolved
    return improvements

def main():
    parser = argparse.ArgumentParser(
        description='Compare two result.json files and find IDs where first fails but second succeeds'
    )
    parser.add_argument('first_file', help='Path to first result.json file')
    parser.add_argument('second_file', help='Path to second result.json file')
    parser.add_argument('--output', '-o', help='Output file (default: stdout)')
    
    args = parser.parse_args()
    
    # Load results from both files
    print(f"Loading results from {args.first_file}...", file=sys.stderr)
    first_resolved, first_unresolved = load_results(args.first_file)
    
    print(f"Loading results from {args.second_file}...", file=sys.stderr)
    second_resolved, second_unresolved = load_results(args.second_file)
    
    # Find improvements
    improvements = find_improvements(first_resolved, first_unresolved, second_resolved, second_unresolved)
    
    # Print summary statistics
    print(f"First file: {len(first_resolved)} resolved, {len(first_unresolved)} unresolved", file=sys.stderr)
    print(f"Second file: {len(second_resolved)} resolved, {len(second_unresolved)} unresolved", file=sys.stderr)
    
    # Prepare output
    output_lines = []
    if improvements:
        print(f"Found {len(improvements)} improvements:", file=sys.stderr)
        for task_id in sorted(improvements):
            output_lines.append(task_id)
            print(f"  {task_id}", file=sys.stderr)
    else:
        print("No improvements found (no cases where first failed but second succeeded)", file=sys.stderr)
    
    # Write output
    if args.output:
        with open(args.output, 'w') as f:
            for line in output_lines:
                f.write(line + '\n')
        print(f"Results written to {args.output}", file=sys.stderr)
    else:
        for line in output_lines:
            print(line)

if __name__ == '__main__':
    main() 