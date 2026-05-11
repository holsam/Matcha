# -- Import external dependencies
import os, sys, typer
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# -- Import internal dependenceis
from matcha.config import load_run_config
from matcha.db import get_connection
from matcha.indexer import run_index
from matcha.matcher import run_match

# -- get_available_configs: returns a dict mapping command name to its parsed config for each file that exists and is valid. Skips malformed files silently
def get_available_configs(matcha_dir: str) -> dict[str, dict[str, Any]]:
    '''Scan for available run configurations in the `.matcha/` directory.'''
    configs = {}
    for command in ('index', 'match'):
        config = load_run_config(matcha_dir, command)
        if config is not None:
            configs[command] = config
    return configs

# -- check_index_stale: queries the database to find the most recent fingerprint timestamp and compares it to the `last_run` timestamp in the match config
def check_index_stale(matcha_dir: str, match_config: dict[str, Any]) -> bool:
    '''Check if the video index has been updated since the last `match` run.'''
    db_path = os.path.join(os.path.dirname(matcha_dir), 'index.db')
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute('SELECT MAX(fingerprinted_at) FROM videos')
        result = cursor.fetchone()
        conn.close()
    except Exception:
        return False
    max_fingerprinted = result[0] if result and result[0] else None
    if max_fingerprinted is None:
        # DB is empty; nothing newer
        return False
    # Parse the last_run ISO 8601 timestamp
    last_run_str = match_config.get('last_run', '')
    # Remove trailing 'Z' if present
    last_run_str = last_run_str.rstrip('Z')
    try:
        last_run = datetime.fromisoformat(last_run_str)
        # Ensure last_run is timezone-aware (UTC)
        if last_run.tzinfo is None:
            last_run = last_run.replace(tzinfo=timezone.utc)
        # Parse max_fingerprinted as a datetime if it's a string
        if isinstance(max_fingerprinted, str):
            max_fp_str = max_fingerprinted.rstrip('Z')
            max_fingerprinted = datetime.fromisoformat(max_fp_str)
            # Ensure it's timezone-aware (UTC)
            if max_fingerprinted.tzinfo is None:
                max_fingerprinted = max_fingerprinted.replace(tzinfo=timezone.utc)
        return max_fingerprinted > last_run
    except (ValueError, AttributeError):
        # If we can't parse timestamps, assume not stale
        return False

# -- invalidate_faiss_index: forces a rebuild of the FAISS index on the next match run 
def invalidate_faiss_index(matcha_dir: str) -> None:
    '''Delete FAISS index files and clear the metadata table.'''
    # Delete FAISS index files
    faiss_file = os.path.join(matcha_dir, 'frame_index.faiss')
    faiss_map_file = os.path.join(matcha_dir, 'frame_index_map.npy')
    for file_path in [faiss_file, faiss_map_file]:
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except OSError:
                pass
    # Clear FAISS metadata table
    db_path = os.path.join(os.path.dirname(matcha_dir), 'index.db')
    try:
        conn = get_connection(db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM faiss_index_meta')
        conn.commit()
        conn.close()
    except Exception:
        # If we can't clear the table, continue anyway
        pass

# -- run_continue: re-runs the last `index` or `match` command using saved configurations
def run_continue(directory: str, command: str | None = None) -> None:
    # Resolve directory to absolute path
    directory = str(Path(directory).resolve())
    matcha_dir = os.path.join(directory, '.matcha')
    db_path = os.path.join(directory, 'index.db')
    # Check that the database exists
    if not os.path.exists(db_path):
        typer.echo(f'Error: No database found at {db_path}. Run `matcha index` first.', err=True)
        raise typer.Exit(code=1)
    # Load available configs
    configs = get_available_configs(matcha_dir)
    if not configs:
        typer.echo('No saved configurations found. Run `matcha index` or `matcha match` first.', err=True)
        raise typer.Exit(code=1)
    # Determine which command to run
    if command is None:
        if len(configs) == 1:
            command = list(configs.keys())[0]
        else:
            # Multiple configs exist; prompt user
            typer.echo('Multiple configurations available:')
            command_list = list(configs.keys())
            for i, cmd in enumerate(command_list, 1):
                typer.echo(f"  {i}. {cmd}")
            choice = typer.prompt("Which command to continue? (1 or 2)")
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(command_list):
                    command = command_list[idx]
                else:
                    typer.echo('Invalid choice.', err=True)
                    raise typer.Exit(code=1)
            except ValueError:
                typer.echo('Invalid choice.', err=True)
                raise typer.Exit(code=1)
    # Check that the requested command has a saved config
    if command not in configs:
        typer.echo(f'Error: No saved configuration for "{command}" command.', err=True)
        raise typer.Exit(code=1)
    config = configs[command]
    args = config.get('args', {})
    # If continuing a match, check if the index is stale
    if command == 'match':
        if check_index_stale(matcha_dir, config):
            typer.echo('Warning: the index has been updated since the last match run. "The FAISS index will be rebuilt before matching.')
            invalidate_faiss_index(matcha_dir)
    # Dispatch to the appropriate command
    if command == 'index':
        run_index(directory, **args)
    elif command == 'match':
        run_match(directory, **args)