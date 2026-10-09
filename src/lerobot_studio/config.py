from __future__ import annotations

import json
import os
from pathlib import Path


def config_dir() -> Path:
    override = os.getenv("LEROBOT_STUDIO_CONFIG")
    path = Path(override).expanduser() if override else Path.home() / ".config" / "lerobot-studio"
    path.mkdir(parents=True, exist_ok=True)
    return path


class DatasetRegistry:
    def __init__(self, path: Path | None = None) -> None:
        self.path = path or config_dir() / "datasets.json"

    def load(self) -> list[Path]:
        if not self.path.exists():
            return []
        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return [Path(item).expanduser().resolve() for item in payload.get("datasets", []) if Path(item).expanduser().exists()]

    def save(self, paths: list[Path]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        unique = list(dict.fromkeys(str(path.resolve()) for path in paths))
        self.path.write_text(json.dumps({"datasets": unique}, indent=2, ensure_ascii=False), encoding="utf-8")

    def add(self, path: Path) -> list[Path]:
        paths = self.load()
        resolved = path.resolve()
        if resolved not in paths:
            paths.append(resolved)
            self.save(paths)
        return paths

    def remove(self, path: Path) -> list[Path]:
        resolved = path.resolve()
        paths = [item for item in self.load() if item != resolved]
        self.save(paths)
        return paths
