#!/bin/bash
# Resumable replay agent runner
# This script can be run in screen/tmux to survive terminal disconnects

cd "$(dirname "$0")"

# Check if there's an existing run to resume from
LATEST_RUN=$(ls -td runs/replay-* 2>/dev/null | head -1)

if [ -n "$LATEST_RUN" ]; then
    echo "📋 Found existing run: $LATEST_RUN"
    echo "💡 To resume, we would skip already completed tasks"
    echo "   (This feature can be added if needed)"
fi

# Run the replay agent
# Use nohup to survive terminal disconnects (but not computer shutdowns)
nohup python3 run_replay_on_savepoints.py --model gpt-5 --n-concurrent 2 > replay_agent.log 2>&1 &

echo "🚀 Replay agent started in background"
echo "📝 Logs: replay_agent.log"
echo "📊 Monitor with: tail -f replay_agent.log"
echo "🔍 Check process: ps aux | grep 'tb run'"


