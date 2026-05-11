"""
Utility functions to save and load CLI argument configurations for `index` and `match` commands, allowing them to be resumed via `matcha continue`
"""

# -- Import external dependencies
import json, os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- save_run_config: returns None, but saves CLI arguments for a command to a JSON config file
def save_run_config(matcha_dir: str, command: str, args: dict[str, Any]) -> None:
    '''Save the CLI arguments for a command to a JSON config file'''
    Path(matcha_dir).mkdir(parents=True, exist_ok=True)
    config = {
        "command": command,
        "last_run": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "args": args,
    }
    config_path = os.path.join(matcha_dir, f"{command}.json")
    temp_path = f"{config_path}.tmp"
    try:
        with open(temp_path, "w") as f:
            json.dump(config, f, indent=2)
        os.replace(temp_path, config_path)
    except Exception as e:
        # Clean up temp file on error
        if os.path.exists(temp_path):
            os.remove(temp_path)
        raise e

# -- load_run_config: returns dictionary of strings and any value, or None if loading saved configuration fails
def load_run_config(matcha_dir: str, command: str) -> dict[str, Any] | None:
    '''Returns the parsed config dict, or None if the file does not exist or contains invalid JSON.'''
    config_path = os.path.join(matcha_dir, f"{command}.json")

    if not os.path.exists(config_path):
        return None

    try:
        with open(config_path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return None