#!/usr/bin/env python3
"""
Simple Python script using Typer to print a list of strings.
"""

import typer
from typing import List, Annotated


def print_list(strings: Annotated[List[str] | None, typer.Option("--list", help="String to add to the list (can be used multiple times)")] = None):
    """
    Print a list of strings.
    
    Args:
        strings: List of strings to print
    """
    if not strings:
        typer.echo("No strings provided. Use --list to add strings.")
        typer.echo("Example: python list_printer.py --list hello --list world --list python")
        return
    
    typer.echo(f"Received list: {strings}")
    typer.echo("Individual items:")
    for i, item in enumerate(strings, 1):
        typer.echo(f"  {i}. {item}")


if __name__ == "__main__":
    typer.run(print_list) 