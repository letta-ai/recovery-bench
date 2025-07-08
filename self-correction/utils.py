import hashlib

def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    return hashlib.sha256(task_description.encode('utf-8')).hexdigest()[:8]
