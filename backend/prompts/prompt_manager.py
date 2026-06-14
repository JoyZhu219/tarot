"""
prompt_manager.py

Loads prompt templates from versioned .txt files using the structure:

  prompts/
  ├── reading_generation/
  │   ├── v1.txt
  │   ├── v2.txt
  │   ├── v3.txt
  │   └── current.txt  ← symlink to active version
  ├── reading_judge/
  │   ├── v1.txt
  │   └── current.txt
  └── config.yaml

Usage:
    from prompts.prompt_manager import prompt_manager

    # Render using current.txt (active version)
    prompt = prompt_manager.render("reading_generation",
                                   user_name="Joy", question="...", ...)

    # Render a specific version explicitly (e.g. for A/B testing)
    prompt = prompt_manager.render("reading_generation", version="v1",
                                   user_name="Joy", ...)

    # Check active version
    prompt_manager.active_version("reading_generation")  # → "v2"

    # List all versions with descriptions
    prompt_manager.list_versions("reading_generation")

    # View usage log
    prompt_manager.usage_log
    # → [{"prompt": "reading_generation", "version": "v2", "at": "..."}]
"""

import re
import yaml
from datetime import datetime, timezone
from pathlib import Path


PROMPTS_DIR = Path(__file__).parent
CONFIG_PATH = PROMPTS_DIR / "config.yaml"


class PromptManager:
    def __init__(self):
        self._config = self._load_config()
        self.usage_log: list[dict] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def active_version(self, prompt_name: str) -> str:
        """Return the active version string for a prompt (e.g. 'v2')."""
        self._assert_prompt_exists(prompt_name)
        return self._config["prompts"][prompt_name]["active"]

    def render(self, prompt_name: str, version: str | None = None, **variables) -> str:
        """
        Load a prompt template, render {{ variable }} placeholders,
        log the usage, and return the final string.

        Args:
            prompt_name: directory name under prompts/ (e.g. "reading_generation")
            version:     e.g. "v1", "v2". None → uses current.txt (active version)
            **variables: template variables to substitute.

        Returns:
            Rendered prompt string.

        Raises:
            ValueError: unknown prompt name or version
            KeyError:   required variable missing from template
        """
        self._assert_prompt_exists(prompt_name)
        resolved_version = version or self.active_version(prompt_name)
        self._assert_version_exists(prompt_name, resolved_version)

        template = self._load_template(prompt_name, version)
        rendered = self._render_template(template, variables)

        self._log_usage(prompt_name, resolved_version)
        return rendered

    def list_versions(self, prompt_name: str) -> list[dict]:
        """Return history list from config.yaml for a prompt."""
        self._assert_prompt_exists(prompt_name)
        return self._config["prompts"][prompt_name].get("history", [])

    def switch_version(self, prompt_name: str, version: str) -> None:
        """
        Update the symlink so current.txt points to the new version.
        Also updates config.yaml active field in memory.
        Note: does NOT persist config.yaml — use update_prompts management
        command for that.
        """
        self._assert_prompt_exists(prompt_name)
        self._assert_version_exists(prompt_name, version)

        current_link = PROMPTS_DIR / prompt_name / "current.txt"
        target = Path(f"{version}.txt")

        if current_link.is_symlink():
            current_link.unlink()
        current_link.symlink_to(target)

        self._config["prompts"][prompt_name]["active"] = version

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_config(self) -> dict:
        if not CONFIG_PATH.exists():
            raise FileNotFoundError(f"Config not found: {CONFIG_PATH}")
        with open(CONFIG_PATH, "r") as f:
            return yaml.safe_load(f)

    def _load_template(self, prompt_name: str, version: str | None) -> str:
        prompt_dir = PROMPTS_DIR / prompt_name
        if version is None:
            # Use current.txt symlink
            path = prompt_dir / "current.txt"
            if not path.exists():
                raise FileNotFoundError(
                    f"current.txt not found in {prompt_dir}. "
                    f"Run: python manage.py update_prompts"
                )
        else:
            path = prompt_dir / f"{version}.txt"
            if not path.exists():
                raise ValueError(
                    f"Template not found: {path}\n"
                    f"Expected: prompts/{prompt_name}/{version}.txt"
                )
        return path.read_text(encoding="utf-8")

    def _render_template(self, template: str, variables: dict) -> str:
        """
        Replace {{ variable_name }} placeholders.
        Raises KeyError if a placeholder has no matching variable.
        """
        def replacer(match):
            key = match.group(1).strip()
            if key not in variables:
                raise KeyError(
                    f"Template placeholder '{{{{ {key} }}}}' has no value. "
                    f"Provided variables: {list(variables.keys())}"
                )
            return str(variables[key])

        return re.sub(r"\{\{\s*(\w+)\s*\}\}", replacer, template)

    def _log_usage(self, prompt_name: str, version: str) -> None:
        self.usage_log.append({
            "prompt": prompt_name,
            "version": version,
            "at": datetime.now(timezone.utc).isoformat(),
        })

    def _assert_prompt_exists(self, prompt_name: str) -> None:
        if prompt_name not in self._config.get("prompts", {}):
            available = list(self._config.get("prompts", {}).keys())
            raise ValueError(
                f"Unknown prompt '{prompt_name}'. Available: {available}"
            )

    def _assert_version_exists(self, prompt_name: str, version: str) -> None:
        history = self._config["prompts"][prompt_name].get("history", [])
        known = [h["version"] for h in history]
        if version not in known:
            raise ValueError(
                f"Unknown version '{version}' for prompt '{prompt_name}'. "
                f"Known versions: {known}"
            )


# Singleton — import and reuse across the Django process
prompt_manager = PromptManager()