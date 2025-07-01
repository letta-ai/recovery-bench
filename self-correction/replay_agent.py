from pathlib import Path
import os
import json
import hashlib
import ast

# or AbstractInstalledAgent if your agent is accessible as a package

from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.agents.terminus import Terminus, Command, CommandBatchResponse
from terminal_bench.llms.chat import Chat
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.base_agent import AgentResult


class ReplayAgent(Terminus):

    @staticmethod
    def name() -> str:
        return "replay-agent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        print(instruction)
        print("HERE!!!!")
        commands, messages = self._read_trajectories(instruction)

        # Replay commands
        self._replay_tasks(session, commands)

        # Create fresh chat and run agent loop
        chat = Chat(self._llm)
        chat._messages = messages
        initial_prompt = session.capture_pane() + "\n\n" + \
            "Previous attempts failed! Please try again with different approaches."
        print(commands, messages)

        self._run_agent_loop(initial_prompt, session, chat, logging_dir)

        return AgentResult(
            total_input_tokens=chat.total_input_tokens,
            total_output_tokens=chat.total_output_tokens,
            failure_mode=FailureMode.NONE,
            timestamped_markers=self._timestamped_markers,
        )

    def _read_episode_response(self, episode_dir: Path) -> CommandBatchResponse | None:
        """Helper method to read and parse response.json from an episode directory."""
        response_file = episode_dir / "response.json"
        if response_file.exists():
            try:
                response_content = response_file.read_text()
                return CommandBatchResponse.model_validate_json(response_content)
            except Exception:
                pass
        return None

    def _read_trajectories(self, task_description: str) -> tuple[list[Command], list[dict]]:
        """Read commands from all episode response.json files and messages from last episode."""
        base_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")
        task_hash = hashlib.sha256(task_description.encode('utf-8')).hexdigest()[:8]
        
        # Find the trajectory folder - it's nested: hash/task_name/task_name.1-of-1/
        trajectory_base = Path(base_folder) / task_hash
        if not trajectory_base.exists():
            return [], []
            
        # Find the nested subdirectory structure
        for subdir in trajectory_base.iterdir():
            if subdir.is_dir():
                for subsubdir in subdir.iterdir():
                    if subsubdir.is_dir() and subsubdir.name.endswith(".1-of-1"):
                        trajectory_folder = subsubdir
                        break
                else:
                    continue
                break
        else:
            return [], []
        
        agent_logs_dir = trajectory_folder / "agent-logs"
        if not agent_logs_dir.exists():
            return [], []
        
        commands = []
        messages = []
        
        # Find all episode directories and sort them numerically
        episode_dirs = []
        for item in agent_logs_dir.iterdir():
            if item.is_dir() and item.name.startswith("episode-"):
                try:
                    episode_num = int(item.name.split("-")[1])
                    episode_dirs.append((episode_num, item))
                except ValueError:
                    continue
        
        # Sort by episode number
        episode_dirs.sort(key=lambda x: x[0])
        
        # Read commands from each episode's response.json using helper method
        for episode_num, episode_dir in episode_dirs:
            parsed_response = self._read_episode_response(episode_dir)
            if parsed_response:
                commands.extend(parsed_response.commands)
        
        # Read messages from the last episode's debug.json and response.json
        if episode_dirs:
            last_episode_dir = episode_dirs[-1][1]
            
            # Read conversation history from debug.json
            debug_file = last_episode_dir / "debug.json"
            if debug_file.exists():
                try:
                    with open(debug_file, 'r') as f:
                        debug_data = json.load(f)
                    if "input" in debug_data and isinstance(debug_data["input"], list):
                        messages = debug_data["input"]
                except Exception:
                    pass
            
            # Add the last assistant response using helper method
            parsed_response = self._read_episode_response(last_episode_dir)
            parsed_response.is_task_complete = False
            if parsed_response:
                assistant_message = {
                    "role": "assistant", 
                    "content": parsed_response.model_dump_json()
                }
                messages.append(assistant_message)
        
        return commands, messages

    def _replay_tasks(self, session: TmuxSession, commands: list[Command]) -> None:
        """Send commands sequentially to session."""
        for command in commands:
            session.send_keys(
                command.keystrokes,
                block=command.is_blocking,
                max_timeout_sec=command.timeout_sec,
            )
