"""Persistent storage for file picker preferences (favorites, recent dirs, last directory)."""

import json
from pathlib import Path

_PREFS_DIR = Path.home() / ".pyopenms-viewer"
_PREFS_FILE = _PREFS_DIR / "file_picker_prefs.json"
_MAX_RECENT = 10


def _load_prefs() -> dict:
    """Load preferences from disk, returning defaults if missing or corrupt."""
    if _PREFS_FILE.exists():
        try:
            with open(_PREFS_FILE) as f:
                data = json.load(f)
            if isinstance(data, dict):
                return data
        except (json.JSONDecodeError, OSError):
            pass
    return {"recent_directories": [], "favorites": [], "last_directory": ""}


def _save_prefs(prefs: dict) -> None:
    """Write preferences to disk."""
    try:
        _PREFS_DIR.mkdir(parents=True, exist_ok=True)
        with open(_PREFS_FILE, "w") as f:
            json.dump(prefs, f, indent=2)
    except OSError:
        pass


def get_last_directory() -> str:
    """Return the last-used directory, or empty string if none stored."""
    return _load_prefs().get("last_directory", "")


def save_last_directory(directory: str) -> None:
    """Persist the last-used directory."""
    prefs = _load_prefs()
    prefs["last_directory"] = directory
    _save_prefs(prefs)


def get_recent_directories() -> list[str]:
    """Return list of recent directories (most recent first), filtering out non-existent paths."""
    prefs = _load_prefs()
    recent = [d for d in prefs.get("recent_directories", []) if Path(d).is_dir()]
    return recent


def add_recent_directory(directory: str) -> None:
    """Add a directory to the recent list (most recent first, max 10)."""
    prefs = _load_prefs()
    recent = prefs.get("recent_directories", [])
    directory = str(Path(directory).resolve())
    if directory in recent:
        recent.remove(directory)
    recent.insert(0, directory)
    prefs["recent_directories"] = recent[:_MAX_RECENT]
    _save_prefs(prefs)


def get_favorites() -> list[str]:
    """Return list of favorite directories, filtering out non-existent paths."""
    prefs = _load_prefs()
    return [d for d in prefs.get("favorites", []) if Path(d).is_dir()]


def add_favorite(directory: str) -> None:
    """Add a directory to favorites."""
    prefs = _load_prefs()
    favs = prefs.get("favorites", [])
    directory = str(Path(directory).resolve())
    if directory not in favs:
        favs.append(directory)
        prefs["favorites"] = favs
        _save_prefs(prefs)


def remove_favorite(directory: str) -> None:
    """Remove a directory from favorites."""
    prefs = _load_prefs()
    favs = prefs.get("favorites", [])
    directory = str(Path(directory).resolve())
    if directory in favs:
        favs.remove(directory)
        prefs["favorites"] = favs
        _save_prefs(prefs)


def is_favorite(directory: str) -> bool:
    """Check if a directory is in favorites."""
    prefs = _load_prefs()
    directory = str(Path(directory).resolve())
    return directory in prefs.get("favorites", [])
