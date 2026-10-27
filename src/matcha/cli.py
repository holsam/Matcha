import os, subprocess, sys
import typer

from matcha.apply import run as run_apply
from matcha.continuer import run_continue
from matcha.db import fold_in_moved_paths, init_schema
from matcha.indexer import run_index
from matcha.matcher import run_match

app = typer.Typer(help="Matcha — a video matching tool using perceptual hashing.")


@app.command(rich_help_panel="Identify matches")
def index(
    directory: str = typer.Argument(..., help="Directory to index."),
    fps: float = typer.Option(1.0, help="Frames per second to sample for hashing."),
    workers: int = typer.Option(4, help="Number of parallel indexing workers."),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio fingerprinting."),
    hwaccel: bool = typer.Option(False, "--hwaccel", help="Use hardware-accelerated decoding (VideoToolbox on Mac)."),
    reuse_duplicates: bool = typer.Option(True, "--reuse-duplicates/--no-reuse-duplicates", help="Reuse stored hashes for byte-identical files instead of decoding them again."),
):
    """Fingerprint all videos in DIRECTORY and store results in .matcha/index.db."""
    run_index(directory, fps=fps, workers=workers, no_audio=no_audio, hwaccel=hwaccel, reuse_duplicates=reuse_duplicates)


@app.command(rich_help_panel="Identify matches")
def match(
    directory: str = typer.Argument(..., help="Directory to match."),
    filter_length: bool = typer.Option(False, "--filter-length", help="Only compare pairs where one video is strictly longer than the other."),
    window: float = typer.Option(10.0, help="Minimum match duration in seconds."),
    frame_step: int = typer.Option(3, help="Step size when sliding the comparison window."),
    threshold: int = typer.Option(10, help="Max Hamming distance to count a frame as matching (0-64)."),
    min_confidence: float = typer.Option(0.8, help="Minimum match ratio to record a result (0.0-1.0)."),
    workers: int = typer.Option(4, help="Number of parallel matching workers."),
    nprobe: int = typer.Option(32, help='FAISS IVF cells to prove (increasing nprobe increases accuracy and runtime).'),
    min_votes: int = typer.Option(2, help='Frame hits that must agree on one offset for a pair to become a candidate (1 keeps every hit).'),
):
    """Compare indexed videos and record matches in .matcha/index.db."""
    run_match(
        directory, 
        filter_length=filter_length,
        window=window,
        frame_step=frame_step,
        threshold=threshold,
        min_confidence=min_confidence,
        workers=workers,
        nprobe=nprobe,
        min_votes=min_votes,
    )



@app.command(rich_help_panel="Review matches")
def review(
    directory: str = typer.Argument(..., help="Directory to review."),
    port: int = typer.Option(8501, help="Port to serve the review UI on."),
):
    """Open the browser UI to review duplicate groups and delete, rename or dismiss videos."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")
    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)
    init_schema(db_path)
    folded = fold_in_moved_paths(db_path)  # videos from the old `move` command are reviewed where they now sit
    if folded:
        typer.echo(f"Adopted the duplicates/ location for {folded} previously moved video(s).")
    app_path = os.path.join(os.path.dirname(__file__), "review", "app.py")
    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", app_path, "--server.port", str(port)],
        env={**os.environ, "MATCHA_REVIEW_DIR": directory},
        check=False,
    )

@app.command(name="apply", rich_help_panel="Review matches")
def apply_(
    directory: str = typer.Argument(..., help="Directory containing .matcha/index.db."),
    plan: str | None = typer.Option(None, "--plan", help="Plan file (default: .matcha/review_plan.json)."),
    execute: bool = typer.Option(False, "--apply", help="Make the changes (default is a dry run)."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    quiet: bool = typer.Option(False, "--quiet", help="Only print the summary and failures."),
):
    """Apply a review plan: delete, rename and update the index db."""
    raise SystemExit(run_apply(directory, plan, apply=execute, yes=yes, quiet=quiet))


@app.command(name="continue", rich_help_panel="Identify matches")
def continue_(
    directory: str = typer.Argument(".", help="Directory to continue from.",),
    command: str | None = typer.Argument(None, help="Command to continue: 'index' or 'match'.",),
):
    """
    Re-run the last index or match command using the same configuration.
    """
    run_continue(directory, command)

if __name__ == "__main__":
    app()