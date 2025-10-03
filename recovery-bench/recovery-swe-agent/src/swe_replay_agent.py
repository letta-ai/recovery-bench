from minisweagent.agents.default import DefaultAgent, AgentConfig
from minisweagent import Model, Environment
from collections.abc import Callable
from minisweagent.agents.default import TerminatingException
from minisweagent.agents.default import NonTerminatingException

#Class for running Agent, DefaultAgent resets self.messages everytime agent.run() is called
class RecoverySWEAgent(DefaultAgent):

    def __init__(self, model: Model, env: Environment, *, config_class: Callable = AgentConfig, messages: list[dict], **kwargs):
        super().__init__(model = model, 
                         env = env, 
                         config_class = config_class, 
                         **kwargs)
        self.messages = messages

    #DefaultAgent instantiates self.messages = [], however we want dirty state message history
    def run(self, task: str, **kwargs) -> tuple[str, str]:
        """Run step() until agent is finished. Return exit status & message"""
        self.extra_template_vars |= {"task": task, **kwargs}
        self.add_message("system", self.render_template(self.config.system_template))
        self.add_message("user", self.render_template(self.config.instance_template))
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
            print(self.messages[-1])
            print("\n\n\n")
