from dataclasses import dataclass
from pathlib import Path


@dataclass
class FilePath:
    path: str | Path

    def __post_init__(self) -> None:
        self.path = self.path.strip()

        try:
            Path(str(self.path))
        except (TypeError, ValueError, OSError) as e:
            raise type(e)(f"Invalid path '{self.path}': {e}") from e

        if not Path(str(self.path)).is_file():
            raise FileNotFoundError(f"File not found at the provided path: '{self.path}'")

        self.path = Path(self.path)
