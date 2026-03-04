import asyncio
import json
import logging
import os
import shlex
import tempfile
import urllib.request
from datetime import datetime
from pathlib import Path

from harbor.agents.installed.base import BaseInstalledAgent, ExecInput
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext

from recovery_bench.utils import calculate_cost, save_usage

logger = logging.getLogger(__name__)

# Keys tried (in order) when extracting agent ID from Letta settings JSON.
_SETTINGS_AGENT_ID_KEYS = ("agent_id", "default_agent_id", "lastAgent", "last_agent")

# Provider keywords used to select the right system prompt for the CLI.
_PROVIDER_SYSTEM_MAP = {
    "letta-claude": ("opus", "sonnet", "haiku", "claude"),
    "letta-codex": ("gpt", "o1-", "o3-"),
    "letta-gemini": ("gemini",),
}


class LettaCode(BaseInstalledAgent):
    """Run Letta Code CLI inside a harbor environment."""

    @staticmethod
    def name() -> str:
        return "letta-code"

    @property
    def _install_agent_template_path(self) -> Path:
        return Path(__file__).parent / "install-letta-code.sh.j2"

    def create_run_agent_commands(self, instruction: str) -> list[ExecInput]:
        # Unused — we override run() directly — but required by the ABC.
        return []

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_agent_id_from_events(events_text: str) -> str | None:
        """Scan JSONL *text* for the first ``agent-*`` id."""
        for line in events_text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            for key in ("agent_id", "session_id"):
                aid = event.get(key)
                if isinstance(aid, str) and aid.startswith("agent-"):
                    return aid
        return None

    @staticmethod
    def _extract_agent_id_from_settings(settings_text: str) -> str | None:
        """Parse Letta ``settings.local.json`` content and return an agent id."""
        if not settings_text.strip():
            return None
        try:
            json_start = settings_text.find("{")
            cleaned = settings_text[json_start:] if json_start != -1 else settings_text
            obj = json.loads(cleaned)
            if not isinstance(obj, dict):
                return None
            for key in _SETTINGS_AGENT_ID_KEYS:
                val = obj.get(key)
                if val:
                    return val
            # Fallback: first value that looks like an agent id.
            for val in obj.values():
                if isinstance(val, str) and val.startswith("agent-"):
                    return val
        except Exception:
            pass
        return None

    @staticmethod
    def _extract_usage_from_events(events_text: str) -> dict:
        """Extract token usage from Letta Code stream-json events.

        Checks two formats:
        1. ``message_type == "usage_statistics"`` events (Letta streaming API)
        2. Last event with ``type == "result"`` containing a ``usage`` field

        Returns a dict with prompt_tokens, completion_tokens,
        cached_input_tokens, cache_write_tokens, reasoning_tokens.
        """
        totals: dict[str, int] = {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "cached_input_tokens": 0,
            "cache_write_tokens": 0,
            "reasoning_tokens": 0,
        }

        parsed_events: list[dict] = []
        found_usage_stats = False

        for line in events_text.splitlines():
            line = line.strip()
            if not line.startswith("{"):
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            parsed_events.append(event)

            if event.get("message_type") == "usage_statistics":
                found_usage_stats = True
                for key in totals:
                    totals[key] += event.get(key) or 0
                details = event.get("prompt_tokens_details") or {}
                totals["cached_input_tokens"] += details.get("cached_tokens") or 0
                details = event.get("completion_tokens_details") or {}
                totals["reasoning_tokens"] += details.get("reasoning_tokens") or 0

        # Fallback: last result event
        if not found_usage_stats and parsed_events:
            last = parsed_events[-1]
            if last.get("type") == "result" and "usage" in last:
                usage = last["usage"]
                for key in totals:
                    totals[key] += usage.get(key) or 0

        return totals

    def _populate_usage(self, events_text: str, context: AgentContext) -> None:
        """Extract cost/usage from *events_text* and populate *context*."""
        model_name = self.model_name or os.environ.get("LETTA_MODEL", "").strip()
        usage = self._extract_usage_from_events(events_text)
        cost = calculate_cost(model_name, usage)
        context.n_input_tokens = usage["prompt_tokens"] or None
        context.n_output_tokens = usage["completion_tokens"] or None
        context.cost_usd = cost if cost > 0 else None

        extra = {}
        for key in ("cached_input_tokens", "cache_write_tokens", "reasoning_tokens"):
            if usage.get(key, 0) > 0:
                extra[key] = usage[key]
        save_usage(self.logs_dir, context, extra_fields=extra or None)

    @staticmethod
    def _build_model_flags(model_name: str) -> str:
        """Return CLI flags for ``--model`` and ``--system``."""
        if not model_name:
            return ""
        flags = f"--model {shlex.quote(model_name)} "
        lower = model_name.lower()
        for system, keywords in _PROVIDER_SYSTEM_MAP.items():
            if any(kw in lower for kw in keywords):
                flags += f"--system {system} "
                break
        return flags

    def _find_events_text(self) -> str:
        """Return events JSONL content from the local logs directory.

        Looks for both the raw download name (``{ts}.events.jsonl``) and
        the renamed copy (``letta_events_{ts}.jsonl``).
        """
        logs_dir = Path(self.logs_dir)
        events_files = sorted(logs_dir.glob("*.events.jsonl")) + sorted(logs_dir.glob("letta_events_*.jsonl"))
        if not events_files:
            return ""
        return events_files[0].read_text()

    # ------------------------------------------------------------------
    # Harbor lifecycle hooks
    # ------------------------------------------------------------------

    def populate_context_post_run(self, context: AgentContext) -> None:
        """Populate agent context from downloaded logs (e.g. after timeout).

        Harbor calls this when ``context.is_empty()`` returns True, which
        happens when ``run()`` is cancelled by a timeout before it can
        populate the context itself.  Harbor's ``_maybe_download_logs``
        copies the container's ``/logs/agent/`` directory to
        ``self.logs_dir`` first, so event files should be available here.
        """
        events_text = self._find_events_text()
        if not events_text.strip():
            return

        agent_id = self._extract_agent_id_from_events(events_text)
        if agent_id:
            (Path(self.logs_dir) / "letta_agent_id_recovered.txt").write_text(agent_id)

        try:
            self._populate_usage(events_text, context)
        except Exception as e:
            logger.warning(f"Failed to extract usage in populate_context_post_run: {e}")

    async def setup(self, environment: BaseEnvironment) -> None:
        """Install the letta CLI inside the task container."""
        await super().setup(environment)

    async def run(
        self,
        instruction: str,
        environment: BaseEnvironment,
        context: AgentContext,
    ) -> None:
        """Invoke letta CLI inside the environment with the given instruction."""

        # --- environment variables ----------------------------------------
        agent_env: dict[str, str] = {}
        for key in ("LETTA_API_KEY", "LETTA_BASE_URL", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
            if key in os.environ:
                agent_env[key] = os.environ[key]

        model_name = self.model_name or os.environ.get("LETTA_MODEL", "").strip()
        if model_name:
            agent_env["LETTA_MODEL"] = model_name

        # --- upload instruction -------------------------------------------
        escaped_instruction = shlex.quote(instruction)
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmpf:
            tmpf.write(instruction)
            local_instr_path = tmpf.name
        try:
            await environment.exec("bash -lc 'mkdir -p /installed-agent'", timeout_sec=None)
            await environment.upload_file(local_instr_path, "/installed-agent/instruction.txt")
        finally:
            try:
                Path(local_instr_path).unlink(missing_ok=True)  # type: ignore[arg-type]
            except Exception:
                pass

        # --- build run script ---------------------------------------------
        ts = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")
        base = f"/logs/agent/{ts}"
        model_flag = self._build_model_flags(model_name)

        run_script = (
            "#!/usr/bin/env bash\n"
            "set -eo pipefail\n"
            "source ~/.bashrc >/dev/null 2>&1 || true\n"
            "mkdir -p /logs/agent\n"
            f"letta --new-agent {model_flag}-p {escaped_instruction} "
            f"--permission-mode bypassPermissions --output-format stream-json "
            f"2>'{base}.stderr.log' | tee '{base}.events.jsonl'\n"
        )

        logs_dir = Path(self.logs_dir)
        logs_dir.mkdir(parents=True, exist_ok=True)
        run_script_path = logs_dir / "run_script.sh"
        run_script_path.write_text(run_script)

        # --- execute ------------------------------------------------------
        result = None
        run_error: Exception | None = None

        async def _capture_settings_after_delay() -> None:
            """Snapshot settings.local.json shortly after the agent starts."""
            try:
                await asyncio.sleep(1.0)
                out = await environment.exec(
                    "bash -lc 'cat .letta/settings.local.json 2>/dev/null || true'",
                    timeout_sec=None,
                )
                mid_agent_id = self._extract_agent_id_from_settings(out.stdout or "")
                if mid_agent_id:
                    (logs_dir / f"letta_agent_id_{ts}_mid.txt").write_text(mid_agent_id)
            except Exception:
                pass

        try:
            await environment.exec("bash -lc 'mkdir -p /installed-agent'", timeout_sec=None)
            tmp_script_path = "/installed-agent/run-letta.sh"
            await environment.upload_file(str(run_script_path), tmp_script_path)
            await environment.exec(f"bash -lc 'chmod +x {tmp_script_path}'", timeout_sec=None)

            asyncio.create_task(_capture_settings_after_delay())

            result = await environment.exec(
                f"bash -lc 'bash {tmp_script_path}'",
                env=agent_env or None,
                timeout_sec=None,
            )
        except Exception as e:
            run_error = e

        # --- collect logs -------------------------------------------------
        agent_id: str | None = None
        events_text: str = ""
        try:
            events_text = await self._download_file(environment, f"{base}.events.jsonl")
            (logs_dir / f"letta_events_{ts}.jsonl").write_text(events_text)

            stderr_text = await self._download_file(environment, f"{base}.stderr.log")
            if stderr_text.strip():
                (logs_dir / f"letta_stderr_{ts}.log").write_text(stderr_text)

            settings_text = await self._download_file(environment, ".letta/settings.local.json")
            agent_id = self._extract_agent_id_from_settings(settings_text)

            if not agent_id:
                agent_id = self._extract_agent_id_from_events(events_text)

            if agent_id:
                (logs_dir / f"letta_agent_id_{ts}.txt").write_text(agent_id)

            if agent_id and run_error is None:
                self._export_agent(agent_id, logs_dir, ts)
        except Exception:
            pass

        # --- populate context ---------------------------------------------
        try:
            self._populate_usage(events_text, context)
        except Exception as e:
            logger.warning(f"Failed to extract/save usage: {e}")

        context.metadata = {
            **(context.metadata or {}),
            "letta_return_code": getattr(result, "return_code", None),
            "letta_logs_ts": ts,
        }

        if run_error is not None:
            raise run_error

    # ------------------------------------------------------------------
    # Private I/O helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def _download_file(environment: BaseEnvironment, remote_path: str) -> str:
        """Cat a file from the environment, returning '' on failure."""
        try:
            out = await environment.exec(
                f"bash -lc 'cat \"{remote_path}\" 2>/dev/null || true'",
                timeout_sec=None,
            )
            return out.stdout or ""
        except Exception:
            return ""

    @staticmethod
    def _export_agent(agent_id: str, logs_dir: Path, ts: str) -> None:
        """Download the ``.af`` agent export (best-effort)."""
        try:
            base_url = os.environ.get("LETTA_BASE_URL", "https://api.letta.com").rstrip("/")
            export_url = f"{base_url}/v1/agents/{agent_id}/export"
            req = urllib.request.Request(export_url, method="GET")
            with urllib.request.urlopen(req, timeout=30) as resp:
                agent_bytes = resp.read()
            (logs_dir / f"letta_agent_export_{ts}.af").write_bytes(agent_bytes)
        except Exception:
            pass
