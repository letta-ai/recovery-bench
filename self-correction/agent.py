from pathlib import Path
from terminal_bench.agents import BaseAgent, AgentResult
# or AbstractInstalledAgent if your agent is accessible as a package
 
from terminal_bench.terminal.tmux_session import TmuxSession
 
class YourCustomAgent(BaseAgent):
    @staticmethod
    def name() -> str:
        return "your-agent-name"
 
    def perform_task(
        self,
        task_description: str,
        session: TmuxSession,
        logging_dir: Path | None = None,
    ) -> AgentResult:
        ...