"""
Class for defining RecoverySWEAgent
"""

from collections.abc import Callable
from minisweagent import Model, Environment
from minisweagent.agents.default import AgentConfig, DefaultAgent
from minisweagent.agents.default import NonTerminatingException
from minisweagent.agents.default import TerminatingException

class RecoverySWEAgent(DefaultAgent):

    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, messages: list[dict], **kwargs):
        """DefaultAgent instantiates self.messages = [], RecoverySWEAgent instantiates self.messages based on context type"""
        super().__init__(model = model, 
                         env = env, 
                         config_class = config_class, 
                         **kwargs)
        self.messages = messages


    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message."""
        self.extra_template_vars |= {"task": task, **kwargs}
        sys_msg = {"role": "system", "content": self.render_template(self.config.system_template)}
        inst_msg = {"role": "user", "content": self.render_template(self.config.instance_template)}

        self.messages = [sys_msg, inst_msg] + self.messages
        i = 0
        while True:
            print(f"Iteration {i} of agent.run")
            try:
                self.step()
            except NonTerminatingException as e:
                self.add_message("user", str(e))
            except TerminatingException as e:
                self.add_message("user", str(e))
                return type(e).__name__, str(e)
            i += 1