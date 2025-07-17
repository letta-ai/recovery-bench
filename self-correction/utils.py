import hashlib
import subprocess

def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode('utf-8')).hexdigest()[:8]

def cleanup_docker():
    """Clean up Docker containers and system resources."""
    print("Cleaning up Docker containers and system resources...")
    
    # Remove all containers (running and stopped)
    try:
        result = subprocess.run(
            "docker rm $(docker ps -aq) -f",
            shell=True,
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Successfully removed Docker containers")
        else:
            print(f"Docker rm command output: {result.stderr}")
    except Exception as e:
        print(f"Error removing Docker containers: {e}")
    
    # Clean up Docker system
    try:
        result = subprocess.run(
            ["docker", "system", "prune", "-f"],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            print("Successfully cleaned up Docker system")
        else:
            print(f"Docker system prune failed: {result.stderr}")
    except Exception as e:
        print(f"Error cleaning up Docker system: {e}")
