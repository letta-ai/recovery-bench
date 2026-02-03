"""
ReplayAgent for Harbor/Terminal-Bench 2.0

This module provides the ReplayAgent for recovery-bench. The agent reads
previous failed trajectories, replays commands to restore the environment
state, and then continues with fresh attempts to complete the task.

Variants:
- ReplayAgent: Full message history recovery
- ReplayAgentWithoutMessages: Environment-only recovery  
- ReplayAgentWithMessageSummaries: Summarized history recovery
"""

from pathlib import Path
import os
import json
from typing import Optional
import asyncio

# Harbor imports
from harbor import BaseAgent, BaseEnvironment, AgentContext
from harbor.agents.terminus_2.tmux_session import TmuxSession
from harbor.models.trial.paths import EnvironmentPaths
from datetime import datetime, timezone
import re

# LiteLLM for LLM calls
import litellm

# Import from utils
from recovery_bench.utils import create_task_hash


class Command:
    """Represents a terminal command to execute."""
    def __init__(self, keystrokes: str, is_blocking: bool = True, timeout_sec: int = 120):
        self.keystrokes = keystrokes
        self.is_blocking = is_blocking
        self.timeout_sec = timeout_sec


class ReplayTerminus(BaseAgent):
    """
    Replay agent that recovers from failed trajectories.
    
    This agent reads a previous failed trajectory, replays all commands to
    restore the environment state, and then continues with the full message
    history, prompting the model to try a different approach.
    """

    def __init__(self, logs_dir=None, model_name: str = "anthropic/claude-sonnet-4-20250514", **kwargs):
        super().__init__(logs_dir=logs_dir, model_name=model_name, **kwargs)
        self._base_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")
        self._include_messages = True
        self._model_name = model_name
        self._max_episodes = 50
        self._messages: list[dict] = []
        self._session: TmuxSession | None = None
        self._trajectory_steps: list[dict] = []
        self._step_counter = 0
        
        # System prompt for the recovery agent
        self._system_prompt = """You are an AI assistant tasked with completing terminal-based tasks.
You have access to a terminal session where you can execute commands.

When given a task:
1. Analyze the current terminal state
2. Plan your approach
3. Execute commands one at a time
4. Check the results before proceeding

Respond with a JSON object containing:
{
    "analysis": "Your analysis of the current state",
    "plan": "Your plan for the next steps", 
    "commands": [{"keystrokes": "command to run", "is_blocking": true, "timeout_sec": 120}],
    "is_task_complete": false
}

Set is_task_complete to true when you believe the task is finished.
"""

    @staticmethod
    def name() -> str:
        return "replay-terminus"

    def version(self) -> str | None:
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent environment with TmuxSession."""
        self._session = TmuxSession(
            session_name=self.name(),
            environment=environment,
            logging_path=EnvironmentPaths.agent_dir / "replay_agent.pane",
            local_asciinema_recording_path=None,
            remote_asciinema_recording_path=None,
        )
        await self._session.start()

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """
        Run the replay agent.
        
        1. Read trajectories from the failed attempt
        2. Replay commands to restore environment state
        3. Continue with agent loop for recovery
        """
        # Read previous trajectories
        commands, messages, n_episodes = self._read_trajectories(instruction)
        
        if len(commands) == 0:
            print("No commands found in trajectory, starting fresh")
            commands = []
            messages = []
            n_episodes = 0

        # Replay commands to restore environment state
        last_output = await self._replay_environment_async(environment, commands)
        print(f"Replayed {len(commands)} commands from previous trajectory")

        # Set up messages for recovery
        self._add_messages(messages)
        
        # Create the recovery prompt
        if self._include_messages:
            recovery_prompt = (
                f"Current terminal state:\n{last_output}\n\n"
                "Previous attempts failed! Please try again with different approaches."
            )
        else:
            recovery_prompt = (
                f"{self._system_prompt}\n\n"
                f"Task: {instruction}\n\n"
                f"Current terminal state:\n{last_output}\n\n"
                "Previous attempts failed! Please try again with different approaches."
            )

        # Record initial user message
        self._add_trajectory_step("user", recovery_prompt)
        
        # Run the agent loop
        await self._run_agent_loop(
            recovery_prompt,
            environment,
            context,
        )
        
        # Save trajectory
        self._save_trajectory()

    def _add_trajectory_step(self, source: str, message: str, tool_calls: list = None) -> None:
        """Add a step to the trajectory."""
        self._step_counter += 1
        step = {
            "step_id": self._step_counter,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "source": source,
            "message": message,
        }
        if tool_calls:
            step["tool_calls"] = tool_calls
        self._trajectory_steps.append(step)
    
    def _save_trajectory(self) -> None:
        """Save trajectory to ATIF format JSON file."""
        trajectory = {
            "schema_version": "ATIF-v1.5",
            "agent": {
                "name": self.name(),
                "version": self.version(),
                "model_name": self._model_name,
            },
            "steps": self._trajectory_steps,
        }
        
        # Use logs_dir passed to agent (not container path)
        if self.logs_dir:
            trajectory_path = Path(self.logs_dir) / "trajectory.json"
        else:
            trajectory_path = Path("trajectory.json")
            
        try:
            trajectory_path.parent.mkdir(parents=True, exist_ok=True)
            with open(trajectory_path, "w") as f:
                json.dump(trajectory, f, indent=2)
            print(f"Trajectory saved to {trajectory_path}")
        except Exception as e:
            print(f"Failed to save trajectory: {e}")

    def _add_messages(self, messages: list[dict]) -> None:
        """Add messages to the conversation history."""
        if self._include_messages:
            self._messages = [{"role": "system", "content": self._system_prompt}]
            self._messages.extend(messages)
        else:
            self._messages = [{"role": "system", "content": self._system_prompt}]

    def _find_trajectory_folder(self, task_hash: str) -> Optional[Path]:
        """Find the trajectory folder based on task hash prefix."""
        base_path = Path(self._base_folder)
        
        if not base_path.exists():
            print(f"Trajectory folder not found: {base_path}")
            return None

        # Look for hash-prefixed directories (format: <hash>-<task-id>/)
        for item in base_path.iterdir():
            if item.is_dir() and item.name.startswith(f"{task_hash}-"):
                # Check agent/ subdirectory (Harbor output structure)
                trajectory_file = item / "agent" / "trajectory.json"
                if trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item
                # Fall back to direct path
                trajectory_file = item / "trajectory.json"
                if trajectory_file.exists():
                    print(f"Found trajectory for hash {task_hash}: {item}")
                    return item

        print(f"No trajectory found for hash {task_hash} in {base_path}")
        return None

    def _read_trajectories(
        self, task_description: str
    ) -> tuple[list[Command], list[dict], int]:
        """Read commands and messages from ATIF trajectory file."""
        task_hash = create_task_hash(task_description)
        print(f"Looking for trajectory with hash: {task_hash}")
        trajectory_folder = self._find_trajectory_folder(task_hash)

        if trajectory_folder is None:
            return [], [], 0

        # Check agent/ subdirectory first (Harbor output structure)
        trajectory_file = trajectory_folder / "agent" / "trajectory.json"
        if not trajectory_file.exists():
            trajectory_file = trajectory_folder / "trajectory.json"
        if not trajectory_file.exists():
            return [], [], 0

        try:
            with open(trajectory_file, "r") as f:
                trajectory = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return [], [], 0

        commands = []
        messages = []
        n_episodes = 0

        # ATIF v1.5 format: steps array with source field
        steps = trajectory.get("steps", trajectory)  # Fall back to trajectory itself if no steps key
        
        for step in steps:
            # Handle ATIF v1.5 format (source/message)
            source = step.get("source", step.get("role", ""))
            content = step.get("message", step.get("content", ""))
            
            if source == "agent" or source == "assistant":
                n_episodes += 1
                # Extract commands from tool_calls (ATIF v1.5)
                tool_calls = step.get("tool_calls", [])
                for tool_call in tool_calls:
                    args = tool_call.get("arguments", {})
                    keystrokes = args.get("keystrokes", "")
                    if keystrokes:
                        commands.append(Command(
                            keystrokes=keystrokes,
                            is_blocking=True,
                            timeout_sec=int(args.get("duration", 1) * 10) + 5  # Convert duration to timeout
                        ))
                
                # Also try parsing message content for commands (old format)
                if not tool_calls:
                    try:
                        response = json.loads(content) if isinstance(content, str) else content
                        if isinstance(response, dict) and "commands" in response:
                            for cmd in response["commands"]:
                                commands.append(Command(
                                    keystrokes=cmd.get("keystrokes", ""),
                                    is_blocking=cmd.get("is_blocking", True),
                                    timeout_sec=cmd.get("timeout_sec", 120)
                                ))
                    except (json.JSONDecodeError, TypeError):
                        pass
            
            # Add to messages for context
            role = "assistant" if source == "agent" else source
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})

        return commands, messages, n_episodes

    async def _replay_environment_async(
        self, environment: BaseEnvironment, commands: list[Command]
    ) -> str:
        """Replay commands in the environment to restore state using TmuxSession."""
        if not self._session:
            print("Warning: TmuxSession not initialized, cannot replay")
            return ""
        
        for command in commands:
            try:
                # Use TmuxSession.send_keys like terminus-2 does
                await self._session.send_keys(
                    keys=command.keystrokes,
                    min_timeout_sec=0.5,
                    max_timeout_sec=float(command.timeout_sec),
                )
            except asyncio.TimeoutError:
                # Continue even on timeout
                continue
            except Exception as e:
                print(f"Replay error: {e}")
                continue

        # Get final terminal state
        try:
            last_output = await self._session.capture_pane()
            return last_output or ""
        except Exception:
            return ""

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run the main agent loop for recovery."""
        
        # Add initial prompt to messages
        self._messages.append({"role": "user", "content": initial_prompt})
        
        for episode in range(self._max_episodes):
            # Call LLM
            try:
                response = await litellm.acompletion(
                    model=self._model_name,
                    messages=self._messages,
                    temperature=0.7,
                )
                assistant_content = response.choices[0].message.content
            except Exception as e:
                print(f"LLM error: {e}")
                break

            # Add assistant response to messages
            self._messages.append({"role": "assistant", "content": assistant_content})
            
            # Log the response
            print(f"Agent response: {assistant_content[:200]}...")
            
            # Record agent step in trajectory
            self._add_trajectory_step("agent", assistant_content)

            # Parse response
            try:
                parsed = json.loads(assistant_content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                json_match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
                if json_match:
                    try:
                        parsed = json.loads(json_match.group())
                    except json.JSONDecodeError:
                        parsed = {"commands": [], "is_task_complete": False}
                else:
                    parsed = {"commands": [], "is_task_complete": False}

            # Check if task is complete (handle both key names)
            if parsed.get("is_task_complete", False) or parsed.get("task_complete", False):
                print("Agent reported task complete")
                break

            # Execute commands using TmuxSession
            commands = parsed.get("commands", [])
            
            for cmd_data in commands:
                keystrokes = cmd_data.get("keystrokes", "")
                duration = cmd_data.get("duration", cmd_data.get("timeout_sec", 1.0))
                
                if keystrokes and self._session:
                    try:
                        await self._session.send_keys(
                            keys=keystrokes,
                            min_timeout_sec=float(duration),
                            max_timeout_sec=180.0,
                        )
                    except Exception as e:
                        print(f"Command error: {e}")

            # Get terminal state after commands
            terminal_output = ""
            if self._session:
                try:
                    terminal_output = await self._session.capture_pane() or ""
                except Exception:
                    pass

            # Log terminal output
            print(f"Terminal output: {terminal_output[:200] if terminal_output else '(empty)'}...")

            # Add terminal output to messages for next iteration (avoid empty content)
            observation = terminal_output.strip() if terminal_output.strip() else "(No output)"
            self._messages.append({"role": "user", "content": observation})
            
            # Record observation in trajectory
            self._add_trajectory_step("user", observation)


class ReplayTerminusWithoutMessages(ReplayTerminus):
    """
    Replay agent that only restores environment state.
    
    This variant replays commands to restore the environment but does NOT
    include the previous conversation history. The model sees only the
    corrupted environment and the original task.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_messages = False

    @staticmethod
    def name() -> str:
        return "replay-terminus-without-messages"


class ReplayTerminusWithMessageSummaries(ReplayTerminus):
    """
    Replay agent that uses summarized message history.
    
    This variant replays commands and provides a summary of the previous
    conversation instead of the full history. This tests whether a
    compressed view of history helps recovery.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._include_messages = True

    @staticmethod
    def name() -> str:
        return "replay-terminus-with-summaries"

    def _add_messages(self, messages: list[dict]) -> None:
        """Create a summary of messages instead of using full history."""
        if not messages:
            self._messages = [{"role": "system", "content": self._system_prompt}]
            return

        # Create a summary using the LLM
        summary_messages = [
            {"role": "system", "content": "You are a helpful assistant that summarizes conversations."},
            {"role": "user", "content": f"Please summarize the following conversation concisely:\n\n{json.dumps(messages, indent=2)}"}
        ]
        
        try:
            # Use sync call for summary (runs during setup)
            response = litellm.completion(
                model=self._model_name,
                messages=summary_messages,
                temperature=0.3,
            )
            summary = response.choices[0].message.content
        except Exception:
            summary = "Previous attempts to complete this task failed."

        self._messages = [
            {"role": "system", "content": self._system_prompt},
            {"role": "assistant", "content": f"Summary of previous attempts:\n{summary}"}
        ]
