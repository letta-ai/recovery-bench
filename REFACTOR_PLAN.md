# Plan: Make recovery-bench easy to evaluate new models & agents

## Problem

Adding a new recovery agent today requires writing ~150 lines of Python that duplicate logic already in `RecoveryTerminus` and `RecoveryLettaCode` — trajectory lookup, ATIF parsing, command replay, and recovery prompting. Harbor already has 10+ installed agents (Claude Code, Codex, Gemini CLI, Aider, Goose, OpenHands, etc.) that should be testable on recovery-bench with near-zero effort.

## Goals

1. **New models**: Already easy via `--recovery-model` with `RecoveryTerminus`. No changes needed.
2. **New agents/harnesses**: Any Harbor installed agent (e.g., `claude-code`) should be wrappable into a recovery agent with **zero code** via a generic CLI flag.
3. **Custom agents**: Advanced users can still write custom recovery agents with full control.

## Architecture

### Core Idea

Extract the shared recovery logic (trajectory lookup → parse → replay → prompt modification) into a reusable module, then provide a generic wrapper (`RecoveryInstalledAgent`) that can wrap any Harbor installed agent at runtime via composition.

### New Files

#### 1. `recovery_bench/replay.py` — Shared replay engine

Extracts and unifies trajectory parsing from `RecoveryTerminus._parse_trajectory()` and `RecoveryLettaCode._parse_atif_trajectory()`. The parsing/normalization is shared; execution engines stay separate.

```python
@dataclass
class ReplayCommand:
    """A command extracted from a previous trajectory for replay."""
    command: str          # Shell command (control sequences stripped)
    keystrokes: str       # Raw keystrokes (for tmux replay)
    timeout_sec: float = 15.0

def find_and_parse_trajectory(logs_dir, base_folder) -> tuple[list[ReplayCommand], list[dict]]:
    """Find trajectory folder for task, parse it, return (commands, messages)."""

def extract_commands(trajectory_folder: Path) -> list[ReplayCommand]:
    """Parse ATIF trajectory and extract commands."""

def extract_messages(trajectory_folder: Path) -> list[dict]:
    """Extract chat messages from trajectory for context injection."""

async def replay_via_exec(environment: BaseEnvironment, commands: list[ReplayCommand]) -> None:
    """Replay commands using environment.exec() — for installed agents."""

async def replay_via_tmux(session, commands: list[ReplayCommand]) -> str:
    """Replay commands using tmux keystrokes — for terminus-based agents."""

def build_recovery_instruction(instruction: str) -> str:
    """Wrap instruction with recovery context."""
```

**DRY scope**: Only trajectory discovery, loading, step normalization, and command/message extraction are shared. The tmux executor (keystrokes + per-command timeouts + pane capture) and exec executor (`environment.exec()` + shell quoting) remain separate since their mechanics differ substantially.

#### 2. `recovery_bench/agents/` — Agent package

```
agents/
  __init__.py       # Public exports + AGENT_REGISTRY
  base.py           # RecoveryInstalledAgent (generic wrapper)
  terminus.py       # RecoveryTerminus variants (moved from recovery_terminus.py)
  letta_code.py     # LettaCode + RecoveryLettaCode (moved from letta_code_agent.py + recovery_letta_code.py)
```

#### 3. `recovery_bench/agents/base.py` — `RecoveryInstalledAgent`

Uses dynamic subclassing via `__new__` — Harbor imports this class, but instantiation returns an instance of a dynamically-created subclass of the wrapped agent. This means the returned object IS-A `ClaudeCode` (or whatever agent), automatically inheriting every method, attribute, `SUPPORTS_ATIF`, `populate_context_post_run()`, cleanup, and any future Harbor additions. No manual delegation needed.

```python
# Cache to avoid creating a new class per instantiation
_recovery_class_cache: dict[type, type] = {}


class RecoveryInstalledAgent:
    """Generic recovery wrapper for any Harbor installed agent.

    Uses __new__ to return a dynamic subclass of the wrapped agent.
    Harbor sees the result as a proper BaseInstalledAgent instance —
    all lifecycle methods (populate_context_post_run, SUPPORTS_ATIF,
    cleanup, etc.) are inherited automatically.

    Usage (via CLI):
        --recovery-agent installed:claude-code
        --recovery-agent installed:codex
        --recovery-agent installed:gemini-cli

    Internally resolved to:
        harbor run --agent-import-path recovery_bench.agents.base:RecoveryInstalledAgent \
                   --agent-kwarg wrapped_agent=claude-code
    """

    def __new__(cls, wrapped_agent: str, **kwargs):
        inner_cls = resolve_harbor_agent(wrapped_agent)

        # Cache dynamic class so type(a) is type(b) for same wrapped agent
        if inner_cls not in _recovery_class_cache:
            class Recovery(inner_cls):
                _trajectory_folder = os.getenv("TRAJECTORY_FOLDER", "./trajectories")

                @staticmethod
                def name() -> str:
                    return f"recovery-{inner_cls.name()}"

                async def setup(self, environment):
                    await super().setup(environment)
                    commands, _ = find_and_parse_trajectory(
                        self.logs_dir, self._trajectory_folder
                    )
                    if commands:
                        await replay_via_exec(environment, commands)

                async def run(self, instruction, environment, context):
                    await super().run(
                        build_recovery_instruction(instruction), environment, context
                    )
                    # Ensure usage.json exists for pipeline aggregation
                    save_usage(self.logs_dir, context)

            Recovery.__name__ = f"Recovery{inner_cls.__name__}"
            Recovery.__qualname__ = f"Recovery{inner_cls.__name__}"
            _recovery_class_cache[inner_cls] = Recovery

        return _recovery_class_cache[inner_cls](**kwargs)
```

**How it works**: `RecoveryInstalledAgent(wrapped_agent="claude-code", logs_dir=..., model_name=...)` returns an instance of `RecoveryClaudeCode(ClaudeCode)`. Harbor's `isinstance(agent, BaseInstalledAgent)` check passes. All lifecycle methods are inherited. The only overrides are `name()`, `setup()` (adds replay after super), and `run()` (wraps instruction + ensures usage accounting).

**Class caching**: Dynamic classes are cached by `inner_cls` so `type(a) is type(b)` holds for agents wrapping the same Harbor agent. No pickle concerns since Harbor doesn't serialize agents.

**Agent resolution**: `resolve_harbor_agent()` uses Harbor's `AgentFactory` / `AgentName` enum to map names like `claude-code` → `ClaudeCode` class. This follows whatever Harbor needs for name validation.

**Usage accounting**: The inner agent populates `AgentContext` (tokens, cost) in its own `run()`. The recovery wrapper calls `save_usage()` after `run()` to ensure `usage.json` is written for pipeline aggregation.

#### 4. `recovery_bench/agents/__init__.py` — Registry

Maps friendly names to import paths. The registry only has the agents we maintain:

```python
AGENT_REGISTRY = {
    # Recovery agents (replay + modified instruction)
    "recovery-terminus": "recovery_bench.agents.terminus:RecoveryTerminus",
    "recovery-terminus-no-messages": "recovery_bench.agents.terminus:RecoveryTerminusWithoutMessages",
    "recovery-terminus-summaries": "recovery_bench.agents.terminus:RecoveryTerminusWithMessageSummaries",
    "recovery-letta-code": "recovery_bench.agents.letta_code:RecoveryLettaCode",

    # Baseline agents (fresh start, no replay)
    "baseline-terminus": "recovery_bench.agents.terminus:BaselineTerminus",

    # Initial agents
    "letta-code": "recovery_bench.agents.letta_code:LettaCode",
}
```

For any Harbor installed agent, the CLI detects the `installed:<name>` syntax and routes through `RecoveryInstalledAgent` automatically — no registry entry needed.

### Changes to Existing Files

#### `generate_traces.py` — Updated CLI

Add support for friendly agent names and the `installed:` prefix:

```bash
# Import paths:
--recovery-agent recovery_bench.agents.terminus:RecoveryTerminus

# Registry names:
--recovery-agent recovery-terminus
--recovery-agent recovery-letta-code

# NEW — wrap any Harbor installed agent:
--recovery-agent installed:claude-code
--recovery-agent installed:codex
--recovery-agent installed:gemini-cli
--recovery-agent installed:aider
```

The `installed:<name>` syntax is resolved in `pipeline.py` to:
```
harbor run --agent-import-path recovery_bench.agents.base:RecoveryInstalledAgent \
           --agent-kwarg wrapped_agent=<name>
```

#### `pipeline.py` — Resolve agent names

Add a `resolve_agent()` function that handles three cases:
1. **Import path** (contains `:`): pass through to `--agent-import-path`
2. **Registry name** (in `AGENT_REGISTRY`): look up import path
3. **`installed:<name>`**: route to `RecoveryInstalledAgent` with `wrapped_agent` kwarg

#### `recovery_terminus.py` → `agents/terminus.py`

Refactor to use `replay.py` for trajectory parsing. Keep the terminus-specific tmux replay and message injection logic but delegate shared parsing to `replay.py`.

#### `recovery_letta_code.py` → `agents/letta_code.py`

Merge with `letta_code_agent.py`. Refactor `RecoveryLettaCode` to use `replay.py`.

#### Usage accounting

Ensure all recovery agents write `usage.json` for pipeline aggregation:
- `RecoveryInstalledAgent`: call `save_usage()` after delegated `run()` completes, using data from `AgentContext`
- `RecoveryTerminus`: already calls `save_usage()` ✓
- `RecoveryLettaCode`: already calls `save_usage()` via `LettaCode._populate_usage()` ✓

### User Experience

#### Evaluating a new model (no code):
```bash
# Same harness (terminus-2), different model
python -m recovery_bench.generate_traces \
    --recovery-model anthropic/claude-opus-4-5-20251101 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

#### Evaluating any Harbor installed agent (no code):
```bash
# Claude Code as recovery agent
python -m recovery_bench.generate_traces \
    --recovery-agent installed:claude-code \
    --recovery-model anthropic/claude-sonnet-4-6 \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859

# Codex as recovery agent
python -m recovery_bench.generate_traces \
    --recovery-agent installed:codex \
    --recovery-model openai/gpt-5.3-codex \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859

# Any other Harbor installed agent — same pattern
python -m recovery_bench.generate_traces \
    --recovery-agent installed:gemini-cli \
    --recovery-model google/gemini-3.1-pro \
    --resume-initial runs/initial-claude-haiku-4-5-20251001-20260303_194859
```

#### Adding a custom agent that's already in Harbor (no code):
If someone has already added `MyAgent` as an installed agent in Harbor (e.g., registered as `my-agent`), they just run:
```bash
python -m recovery_bench.generate_traces \
    --recovery-agent installed:my-agent \
    --recovery-model <model> \
    --resume-initial <traces>
```
No Python wrapper needed — `RecoveryInstalledAgent` handles it at runtime.

## Implementation Order

1. **`replay.py`** — Extract shared replay logic (trajectory parsing, command extraction, instruction wrapping, separate executors)
2. **`agents/base.py`** — `RecoveryInstalledAgent` generic wrapper (dynamic subclassing via `__new__`, class caching)
3. **`agents/terminus.py`** — Move + refactor `RecoveryTerminus` to use `replay.py`
4. **`agents/letta_code.py`** — Move + merge `LettaCode` + `RecoveryLettaCode`, refactor to use `replay.py`
5. **`agents/__init__.py`** — Registry and exports
6. **`generate_traces.py` + `pipeline.py`** — Update CLI to resolve friendly names + `installed:` prefix
7. **`__init__.py`** — Update top-level exports
8. **Clean up** — Remove old `recovery_terminus.py`, `recovery_letta_code.py`, `letta_code_agent.py`
9. **README.md** — Update documentation

## Constraints & Scope

- **Installed agents are recovery-only**: `RecoveryInstalledAgent` is only used for recovery runs, not initial trace generation. Initial traces always use terminus-2 (or a custom initial agent).
- **Single iteration recovery**: No multi-iteration recovery for installed agents. The trajectory parser only handles ATIF format (terminus-2 trajectories). This is fine because initial traces come from terminus-2.
- **No extra agent kwargs passthrough**: Wrapped agent config comes from model configs (JSON files via `--recovery-model`). Users who need custom agent kwargs can create new configs.

## Notes

- `RecoveryTerminus` stays special because it uses tmux replay (keystrokes, not shell commands) and supports message history injection — these are terminus-specific features that don't apply to installed agents.
- `RecoveryInstalledAgent` uses dynamic subclassing via `__new__` — the returned instance IS-A the wrapped agent class, so all Harbor lifecycle methods (`populate_context_post_run()`, `SUPPORTS_ATIF`, `create_cleanup_commands()`, and any future additions) are inherited automatically. No manual delegation.
- `isinstance(agent, RecoveryInstalledAgent)` is `False` (the instance is a dynamic subclass of the wrapped agent, not of `RecoveryInstalledAgent`). This is fine — we don't need that check. Use marker attribute if needed.
- Dynamic classes are cached per `inner_cls` for identity stability. Pickle not supported but not needed.
- `resolve_harbor_agent()` follows Harbor's own agent name resolution (via `AgentName` enum / `AgentFactory`) to stay compatible with Harbor updates.
