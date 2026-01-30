"""
ReplayAgent for Harbor/Terminal-Bench 2.0

This module provides replay agents that can recover from failed trajectories.
The agents read previous failed trajectories, replay the commands to restore
the environment state, and then continue with fresh attempts.

Three variants are provided:
- ReplayAgent: Full message history recovery
- ReplayAgentWithoutMessages: Environment-only recovery  
- ReplayAgentWithMessageSummaries: Summarized history recovery
"""

from pathlib import Path
import os
import json
from abc import ABC, abstractmethod
from typing import Optional
import asyncio

# Harbor imports
from harbor.agents.base import BaseAgent
from harbor.models.environment.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

# LiteLLM for LLM calls
import litellm


def create_task_hash(task_description: str) -> str:
    """Create 8-character hash from task description."""
    import hashlib
    return hashlib.sha256(task_description.encode("utf-8")).hexdigest()[:8]


class Command:
    """Represents a terminal command to execute."""
    def __init__(self, keystrokes: str, is_blocking: bool = True, timeout_sec: int = 120):
        self.keystrokes = keystrokes
        self.is_blocking = is_blocking
        self.timeout_sec = timeout_sec


class ReplayABC(ABC):
    """Abstract base class for replay functionality."""
    
    @abstractmethod
    def _replay_environment(self, environment: BaseEnvironment, commands: list[Command]) -> str:
        pass

    @abstractmethod
    def _add_messages(self, messages: list[dict]) -> None:
        pass

    @abstractmethod
    def _read_trajectories(
        self, task_description: str, logging_dir: Optional[Path] = None
    ) -> tuple[list[Command], list[dict], int]:
        pass


class ReplayAgent(BaseAgent, ReplayABC):
    """
    Replay agent that recovers from failed trajectories.
    
    This agent reads a previous failed trajectory, replays all commands to
    restore the environment state, and then continues with the full message
    history, prompting the model to try a different approach.
    """

    def __init__(self, model_name: str = "anthropic/claude-sonnet-4-20250514", **kwargs):
        self._base_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")
        self._include_messages = True
        self._model_name = model_name
        self._max_episodes = 50
        self._messages: list[dict] = []
        
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
        return "replay-agent"

    def version(self) -> str | None:
        return "2.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        """Setup the agent environment."""
        pass

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
            context.add_log("No commands found in trajectory, starting fresh")
            commands = []
            messages = []
            n_episodes = 0

        # Replay commands to restore environment state
        last_output = await self._replay_environment_async(environment, commands)
        context.add_log(f"Replayed {len(commands)} commands from previous trajectory")

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

        # Run the agent loop
        await self._run_agent_loop(
            recovery_prompt,
            instruction,
            environment,
            context,
            n_episodes
        )

    def _add_messages(self, messages: list[dict]) -> None:
        """Add messages to the conversation history."""
        if self._include_messages:
            self._messages = [{"role": "system", "content": self._system_prompt}]
            self._messages.extend(messages)
        else:
            self._messages = [{"role": "system", "content": self._system_prompt}]

    def _find_trajectory_folder(self, task_hash: str) -> Optional[Path]:
        """Find the trajectory folder based on task hash (ATIF format)."""
        base_path = Path(self._base_folder)
        
        if not base_path.exists():
            return None

        # Look for task directories with trajectory.json
        for item in base_path.iterdir():
            if item.is_dir():
                trajectory_file = item / "trajectory.json"
                if trajectory_file.exists():
                    return item

        return None

    def _read_trajectories(
        self, task_description: str, logging_dir: Optional[Path] = None
    ) -> tuple[list[Command], list[dict], int]:
        """Read commands and messages from ATIF trajectory file."""
        task_hash = create_task_hash(task_description)
        trajectory_folder = self._find_trajectory_folder(task_hash)

        if trajectory_folder is None:
            return [], [], 0

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

        for step in trajectory:
            role = step.get("role", "")
            content = step.get("content", "")
            
            if role == "assistant":
                n_episodes += 1
                # Parse command batch from assistant response
                try:
                    response = json.loads(content) if isinstance(content, str) else content
                    if "commands" in response:
                        for cmd in response["commands"]:
                            commands.append(Command(
                                keystrokes=cmd.get("keystrokes", ""),
                                is_blocking=cmd.get("is_blocking", True),
                                timeout_sec=cmd.get("timeout_sec", 120)
                            ))
                except (json.JSONDecodeError, TypeError):
                    pass
            
            # Add to messages for context
            if role in ["user", "assistant", "system"]:
                messages.append({"role": role, "content": content})

        return commands, messages, n_episodes

    def _replay_environment(self, environment: BaseEnvironment, commands: list[Command]) -> str:
        """Synchronous wrapper for replay (for compatibility)."""
        return asyncio.get_event_loop().run_until_complete(
            self._replay_environment_async(environment, commands)
        )

    async def _replay_environment_async(
        self, environment: BaseEnvironment, commands: list[Command]
    ) -> str:
        """Replay commands in the environment to restore state."""
        last_output = ""
        
        for command in commands:
            try:
                result = await environment.exec(
                    command=command.keystrokes,
                    timeout_sec=command.timeout_sec
                )
                last_output = result.stdout if result else ""
            except asyncio.TimeoutError:
                # Continue even on timeout (like old behavior)
                continue
            except Exception as e:
                # Log but continue
                continue

        # Capture final terminal state
        try:
            result = await environment.exec("tmux capture-pane -p")
            last_output = result.stdout if result else last_output
        except Exception:
            pass

        return last_output

    async def _run_agent_loop(
        self,
        initial_prompt: str,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
        start_episode: int = 0
    ) -> None:
        """Run the main agent loop for recovery."""
        
        # Add initial prompt to messages
        self._messages.append({"role": "user", "content": initial_prompt})
        
        for episode in range(start_episode, start_episode + self._max_episodes):
            # Call LLM
            try:
                response = await litellm.acompletion(
                    model=self._model_name,
                    messages=self._messages,
                    temperature=0.7,
                )
                assistant_content = response.choices[0].message.content
            except Exception as e:
                context.add_log(f"LLM error: {e}")
                break

            # Add assistant response to messages
            self._messages.append({"role": "assistant", "content": assistant_content})
            
            # Log the response
            context.add_step(
                action=assistant_content,
                observation="",
            )

            # Parse response
            try:
                parsed = json.loads(assistant_content)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                try:
                    import re
                    json_match = re.search(r'\{.*\}', assistant_content, re.DOTALL)
                    if json_match:
                        parsed = json.loads(json_match.group())
                    else:
                        parsed = {"commands": [], "is_task_complete": False}
                except:
                    parsed = {"commands": [], "is_task_complete": False}

            # Check if task is complete
            if parsed.get("is_task_complete", False):
                context.add_log("Agent reported task complete")
                break

            # Execute commands
            commands = parsed.get("commands", [])
            terminal_output = ""
            
            for cmd_data in commands:
                keystrokes = cmd_data.get("keystrokes", "")
                timeout = cmd_data.get("timeout_sec", 120)
                
                try:
                    result = await environment.exec(
                        command=keystrokes,
                        timeout_sec=timeout
                    )
                    terminal_output = result.stdout if result else ""
                except asyncio.TimeoutError:
                    terminal_output = "[Command timed out]"
                except Exception as e:
                    terminal_output = f"[Error: {e}]"

            # Capture terminal state
            try:
                result = await environment.exec("tmux capture-pane -p")
                terminal_output = result.stdout if result else terminal_output
            except Exception:
                pass

            # Update context with observation
            if context._steps:
                context._steps[-1].observation = terminal_output

            # Add terminal output to messages for next iteration
            self._messages.append({"role": "user", "content": terminal_output})


class ReplayAgentWithoutMessages(ReplayAgent):
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
        return "replay-agent-without-messages"


class ReplayAgentWithMessageSummaries(ReplayAgent):
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
        return "replay-agent-with-summaries"

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
