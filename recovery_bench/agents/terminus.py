"""BaselineTerminus - Runs Terminus2 fresh with no recovery replay.

Used to compare "strong model from scratch" vs "strong model with recovery".
"""

from harbor.agents.terminus_2.terminus_2 import Terminus2
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.utils import save_usage


class BaselineTerminus(Terminus2):
    """Baseline agent: runs Terminus2 fresh on the task with no replay.

    Used to compare "strong model from scratch" vs "strong model with recovery".
    Accepts model_kwargs so it works with the same pipeline.
    """

    def __init__(self, model_kwargs: dict = None, **kwargs):
        super().__init__(**(model_kwargs or {}), **kwargs)

    @staticmethod
    def name() -> str:
        return "baseline-terminus"

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Run Terminus2, then save usage.json for aggregation."""
        await super().run(instruction, environment, context)
        save_usage(self.logs_dir, context)
