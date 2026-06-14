"""
update_prompts.py

Management command to sync current.txt symlinks with config.yaml.

Run after changing "active" version in config.yaml:
    python manage.py update_prompts

Or switch a specific prompt:
    python manage.py update_prompts --prompt reading_generation --version v3
"""

from django.core.management.base import BaseCommand
from prompts.prompt_manager import prompt_manager


class Command(BaseCommand):
    help = "Sync current.txt symlinks with active versions in config.yaml"

    def add_arguments(self, parser):
        parser.add_argument("--prompt", type=str, default=None,
                            help="Specific prompt to update (default: all)")
        parser.add_argument("--version", type=str, default=None,
                            help="Version to switch to (requires --prompt)")

    def handle(self, *args, **options):
        target_prompt = options["prompt"]
        target_version = options["version"]

        if target_version and not target_prompt:
            self.stderr.write("--version requires --prompt")
            return

        prompts = (
            [target_prompt]
            if target_prompt
            else list(prompt_manager._config["prompts"].keys())
        )

        for prompt_name in prompts:
            version = target_version or prompt_manager.active_version(prompt_name)
            try:
                prompt_manager.switch_version(prompt_name, version)
                self.stdout.write(
                    self.style.SUCCESS(f"  ✓ {prompt_name} → {version}.txt")
                )
            except Exception as e:
                self.stderr.write(f"  ✗ {prompt_name}: {e}")

        self.stdout.write("\nDone. Current symlinks:")
        for prompt_name in prompts:
            from pathlib import Path
            link = Path(__file__).parent.parent.parent.parent / "prompts" / prompt_name / "current.txt"
            if link.is_symlink():
                self.stdout.write(f"  {prompt_name}/current.txt → {link.resolve().name}")