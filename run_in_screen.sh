#!/bin/bash
# Run replay agent in a screen session for terminal disconnect resilience
# This will survive terminal disconnects (but NOT computer shutdowns)

cd "$(dirname "$0")"

SESSION_NAME="replay-agent"

# Check if screen session already exists
if screen -list | grep -q "$SESSION_NAME"; then
    echo "⚠️  Screen session '$SESSION_NAME' already exists!"
    echo "   Attach with: screen -r $SESSION_NAME"
    echo "   Or kill it first: screen -S $SESSION_NAME -X quit"
    exit 1
fi

# Start screen session and run the replay agent
echo "🚀 Starting replay agent in screen session: $SESSION_NAME"
echo "📋 To attach: screen -r $SESSION_NAME"
echo "📋 To detach: Press Ctrl+A then D"
echo "📋 To kill: screen -S $SESSION_NAME -X quit"

screen -dmS "$SESSION_NAME" bash -c "python3 run_replay_on_savepoints.py --model gpt-5 --n-concurrent 2; exec bash"

echo ""
echo "✅ Screen session started!"
echo "💡 Run 'screen -r $SESSION_NAME' to view progress"


