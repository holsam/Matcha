import os, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from rich.console import Console, Group
from rich.live import Live
from rich.progress import (
    BarColumn, MofNCompleteColumn, Progress,
    SpinnerColumn, TaskProgressColumn, TimeRemainingColumn,
)
from rich.text import Text


from .db import get_connection, init_schema
from .fingerprint import (
    VIDEO_EXTENSIONS,
    extract_frame_hashes,
    get_audio_fingerprint,
    get_video_duration,
)

console = Console()

_local = threading.local()
_id_lock = threading.Lock()
_id_counter = 0
_worker_status: dict[int, str | None] = {}
_status_lock = threading.Lock()
_stop_event = threading.Event()

def _reset_worker_state():
    global _id_counter, _worker_status
    _id_counter = 0
    _worker_status = {}
    _stop_event.clear()

def _get_worker_id() -> int:
    global _id_counter
    if not hasattr(_local, "worker_id"):
        with _id_lock:
            _id_counter += 1
            _local.worker_id = _id_counter
    return _local.worker_id

def _set_status(filename: str | None):
    with _status_lock:
        _worker_status[_get_worker_id()] = filename

def _make_progress() -> Progress:
    return Progress(
        SpinnerColumn(),
        "[progress.description]{task.description}",
        BarColumn(), MofNCompleteColumn(),
        TaskProgressColumn(), TimeRemainingColumn(),
        console=console,
    )

def _render(progress: Progress, num_workers: int) -> Group:
    """Progress bar + one dim status line per worker."""
    with _status_lock:
        snapshot = dict(_worker_status)
    quitting = _stop_event.is_set()
    lines: list = [progress]
    for i in range(1, num_workers + 1):
        if quitting:
            label = "quitting"
        else:
            label = snapshot.get(i) or "idle"
        lines.append(Text(f"  Worker {i}: {label}", style="dim"))
    return Group(*lines)

_thread_local_conn = threading.local()

def _get_conn(db_path: str):
    if not hasattr(_thread_local_conn, "conn"):
        _thread_local_conn.conn = get_connection(db_path)
    return _thread_local_conn.conn

def _rollback_video(conn, video_id: int):
    """Remove all partial data written for a video that was interrupted mid-processing."""
    with conn:
        conn.execute("DELETE FROM frame_hashes WHERE video_id = ?", (video_id,))
        conn.execute("DELETE FROM audio_fingerprints WHERE video_id = ?", (video_id,))
        conn.execute(
            "UPDATE videos SET duration = NULL, fingerprinted_at = NULL WHERE id = ?",
            (video_id,),
        )

def find_videos(root: str) -> list[str]:
    videos = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d != ".matcha"]
        for fname in filenames:
            if Path(fname).suffix.lower() in VIDEO_EXTENSIONS:
                videos.append(os.path.join(dirpath, fname))
    return sorted(videos)

def register_videos(db_path: str, paths: list[str]):
    conn = get_connection(db_path)
    with conn:
        conn.executemany(
            "INSERT OR IGNORE INTO videos (path) VALUES (?)",
            [(p,) for p in paths],
        )

def get_unprocessed(db_path: str) -> list[tuple[int, str]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT id, path FROM videos WHERE fingerprinted_at IS NULL"
    ).fetchall()
    return [(row["id"], row["path"]) for row in rows]

def process_video(args: tuple) -> tuple[str, str | None]:
    """
    Worker — runs in a thread. Sets its status line before and after
    processing so the Live display reflects what each worker is doing.
    Returns (video_path, error_message).
    """
    video_id, video_path, db_path, fps, no_audio, hwaccel = args
    _set_status(os.path.basename(video_path))
    conn = _get_conn(db_path)

    try:
        if _stop_event.is_set():
            return video_path, "cancelled"

        duration = get_video_duration(video_path)

        if _stop_event.is_set():
            return video_path, "cancelled"

        frame_hashes = extract_frame_hashes(video_path, fps=fps, hwaccel=hwaccel)

        if _stop_event.is_set():
            return video_path, "cancelled"

        audio = None if no_audio else get_audio_fingerprint(video_path)

        if _stop_event.is_set():
            return video_path, "cancelled"

        with conn:
            conn.execute(
                "UPDATE videos SET duration = ? WHERE id = ?",
                (duration, video_id),
            )
            conn.executemany(
                "INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)",
                [(video_id, ts, ph) for ts, ph in frame_hashes],
            )
            if audio is not None:
                audio_duration, fingerprint = audio
                conn.execute(
                    """
                    INSERT OR REPLACE INTO audio_fingerprints
                        (video_id, duration, fingerprint)
                    VALUES (?, ?, ?)
                    """,
                    (video_id, audio_duration, fingerprint),
                )
            conn.execute(
                "UPDATE videos SET fingerprinted_at = ? WHERE id = ?",
                (time.time(), video_id),
            )
        _set_status(None)
        return video_path, None

    except Exception as e:
        _rollback_video(conn, video_id)
        _set_status(None)
        return video_path, str(e)


def _listen_for_quit():
    """
    Background thread that sets _stop_event when 'q' is pressed.
    Works on Mac/Linux via termios, and Windows via msvcrt.
    Degrades silently if neither is available.
    """
    try:
        if sys.platform == "win32":
            import msvcrt
            while not _stop_event.is_set():
                if msvcrt.kbhit():
                    ch = msvcrt.getwch()
                    if ch.lower() == "q":
                        _stop_event.set()
                        break
                time.sleep(0.05)
        else:
            import termios, tty
            fd = sys.stdin.fileno()
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setraw(fd)
                while not _stop_event.is_set():
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        _stop_event.set()
                        break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass  # Not a TTY or unsupported platform — quit-on-q simply unavailable


def run_index(
    directory: str,
    fps: float = 1.0,
    workers: int = 4,
    no_audio: bool = False,
    hwaccel: bool = False,
):
    _reset_worker_state()
    directory = os.path.abspath(directory)
    db_dir = os.path.join(directory, ".matcha")
    os.makedirs(db_dir, exist_ok=True)
    db_path = os.path.join(db_dir, "index.db")

    init_schema(db_path)

    console.print(f"\n:tea: [bold green]Matcha[/bold green]")
    console.print(f"Scanning [cyan]{directory}[/cyan] for videos...")
    all_videos = find_videos(directory)
    time.sleep(1.5)
    console.print(f"Found {len(all_videos)} video(s).")

    register_videos(db_path, all_videos)

    to_process = get_unprocessed(db_path)
    console.print(f'{len(to_process)} video(s) to index.\n')
    if not to_process:
        console.print(f"[bold green]All videos already indexed![/bold green]\n")
        return
    
    console.print(f"Indexing configuration: {workers} worker(s), {fps}fps, hardware acceleration = {hwaccel}, audio fingerprinting = {not no_audio}"
    )
    console.print(f"[dim]Press [bold]q[/bold] to quit indexing before completion.[/dim]\n")

    args = [
        (vid_id, path, db_path, fps, no_audio, hwaccel)
        for vid_id, path in to_process
    ]

    quit_thread = threading.Thread(target=_listen_for_quit, daemon=True)
    quit_thread.start()

    errors: list[str] = []
    progress = _make_progress()
    task = progress.add_task("Indexing", total=len(args))

    with Live(_render(progress, workers), console=console, refresh_per_second=10) as live:
        executor = ThreadPoolExecutor(max_workers=workers)
        try:
            futures = {executor.submit(process_video, arg): arg for arg in args}
            pending = set(futures.keys())
            while pending:
                done = {f for f in pending if f.done()}
                for future in done:
                    pending.remove(future)
                    path, error = future.result()
                    if error != "cancelled":
                        progress.advance(task)
                    if error:
                        errors.append(f"{path}: {error}")

                if _stop_event.is_set():
                    for f in pending:
                       f.cancel()
                    live.update(_render(progress, workers))
                    time.sleep(0.5)
                    break

                live.update(_render(progress, workers))
                time.sleep(0.05)

        finally:
            executor.shutdown(wait=False, cancel_futures=True)

    if _stop_event.is_set():
        time.sleep(0.5)
        console.print(f"\n[yellow]Indexing cancelled.[/yellow]\n")
        sys.exit()

    if errors:
        console.print(f"\n[red]{len(errors)} video(s) failed:[/red]")
        for msg in errors:
            console.print(f"  [dim][SKIP] {msg}[/dim]")

    console.print(f"[bold green]Indexing complete.[/bold green]\n")