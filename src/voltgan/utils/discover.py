import re
from datetime import datetime
from pathlib import Path

_DATETIME_PATTERN = re.compile(r"(\d{4}-\d{2}-\d{2})[_ ](\d{2}\.\d{2}\.\d{2})")


class FileDiscoverer:
    @staticmethod
    def find(root_dir: Path, subdirs: list[str], exts: tuple[str, ...]) -> list[Path]:
        paths = []
        for subdir in subdirs:
            target = root_dir / subdir
            if target.exists():
                paths.extend(
                    [
                        p
                        for p in target.rglob("*")
                        if p.is_file() and p.suffix.lower() in exts
                    ]
                )
        return paths

    @staticmethod
    def sort_wuppertal(path: Path) -> datetime:
        m = _DATETIME_PATTERN.search(path.name)
        if not m:
            raise ValueError(f"Cannot parse date-time from filename: {path.name}")
        return datetime.strptime(
            f"{m.group(1)} {m.group(2).replace('.', ':')}", "%Y-%m-%d %H:%M:%S"
        )
