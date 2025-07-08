from pathlib import Path
import os
import json
import shutil

# or AbstractInstalledAgent if your agent is accessible as a package

from terminal_bench.terminal.tmux_session import TmuxSession
from terminal_bench.agents.terminus import Terminus, Command, CommandBatchResponse
from terminal_bench.llms.chat import Chat
from terminal_bench.agents.failure_mode import FailureMode
from terminal_bench.agents.base_agent import AgentResult
from .utils import create_task_hash

class ReplayAgent(Terminus):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    @staticmethod
    def name() -> str:
        return "replay-agent"

    def perform_task(
        self,
        instruction: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:

        commands, messages = self._read_trajectories(instruction, logging_dir)

        # Replay commands
        last_pane_output = self._replay_tasks(session, commands)

        # Create fresh chat and run agent loop
        chat = Chat(self._llm)
        chat._messages = messages
        initial_prompt = last_pane_output + "\n\n" + \
            "Previous attempts failed! Please try again with different approaches."
        # print(initial_prompt)
        # print(commands, messages)

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

    def _read_trajectories(self, task_description: str, logging_dir: Path | None = None) -> tuple[list[list[Command]], list[dict]]:
        """Read commands from all episode response.json files and messages from last episode."""
        base_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")
        task_hash = create_task_hash(task_description)
        
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

        if logging_dir is not None:
            # copy all the files in the agent-logs directory to the logging_dir, including the subdirectories
            shutil.copytree(agent_logs_dir, logging_dir / "agent-logs")

            # create a file indicting how many episodes have been replayed
            (logging_dir / "episodes_replayed.txt").write_text(f"replayed {len(episode_dirs)} episodes from {agent_logs_dir}")

        # Sort by episode number
        episode_dirs.sort(key=lambda x: x[0])
        
        # Read commands from each episode's response.json using helper method
        for episode_num, episode_dir in episode_dirs:
            parsed_response = self._read_episode_response(episode_dir)
            if parsed_response:
                commands.append(parsed_response.commands)
        
        last_episode_dir = episode_dirs[-1][1]
        
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

    def _replay_tasks(self, session: TmuxSession, commands: list[list[Command]]) -> None:
        """Send commands sequentially to session."""
        for command_list in commands:
            _, result = self._execute_commands(command_list, session)
        return result
    
    def _run_agent_loop(
        self,
        initial_prompt: str,
        session: TmuxSession,
        chat: Chat,
        n_episodes: int,
        logging_dir: Path | None = None,
    ) -> None:
        """n_episodes is the starting episode number (episode so far)"""
        prompt = initial_prompt

        for episode in range(n_episodes, n_episodes + self._max_episodes):
            logging_paths = self._setup_episode_logging(logging_dir, episode)

            parsed_response = self._handle_llm_interaction(chat, prompt, logging_paths)
            self._record_asciinema_marker(parsed_response.model_dump_json(), session)

            timeout_occurred, terminal_output = self._execute_commands(
                parsed_response.commands, session
            )

            if parsed_response.is_task_complete:
                break

            prompt = terminal_output
