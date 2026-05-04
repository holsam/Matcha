import typer

from .indexer import run_index
from .matcher import run_match
from .mover import run_move
from .cleanup import run_cleanup
from .sync import run_sync

app = typer.Typer(help="Matcha — a video matching tool using perceptual hashing.")


@app.command()
def index(
    directory: str = typer.Argument(..., help="Directory to index."),
    fps: float = typer.Option(1.0, help="Frames per second to sample for hashing."),
    workers: int = typer.Option(4, help="Number of parallel indexing workers."),
    no_audio: bool = typer.Option(False, "--no-audio", help="Skip audio fingerprinting."),
    hwaccel: bool = typer.Option(False, "--hwaccel", help="Use hardware-accelerated decoding (VideoToolbox on Mac)."),
):
    """Fingerprint all videos in DIRECTORY and store results in .matcha/index.db."""
    run_index(directory, fps=fps, workers=workers, no_audio=no_audio, hwaccel=hwaccel)


@app.command()
def match(
    directory: str = typer.Argument(..., help="Directory to match."),
    filter_length: bool = typer.Option(False, "--filter-length", help="Only compare pairs where one video is strictly longer than the other."),
    window: float = typer.Option(10.0, help="Minimum match duration in seconds."),
    frame_step: int = typer.Option(3, help="Step size when sliding the comparison window."),
    threshold: int = typer.Option(10, help="Max Hamming distance to count a frame as matching (0-64)."),
    min_confidence: float = typer.Option(0.8, help="Minimum match ratio to record a result (0.0-1.0)."),
    workers: int = typer.Option(4, help="Number of parallel matching workers."),
    nprobe: int = typer.Option(32, help='FAISS IVF cells to prove (increasing nprobe increases accuracy and runtime).')
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
        nprobe=nprobe
    )


@app.command()
def move(
    directory: str = typer.Argument(..., help="Directory to organise."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without moving any files."),
):
    """Move matched videos into sequentially numbered subdirectories under duplicates/."""
    run_move(directory, dry_run=dry_run)

@app.command()
def cleanup(
    directory: str = typer.Argument(..., help="Directory to clean up."),
):
    """
    Check duplicates/ for files deleted by the user since moving.

    Removes deleted files from the index. If only one file remains in a
    subdirectory, returns it to its original location.
    """
    run_cleanup(directory)


@app.command()
def sync(
    directory: str = typer.Argument(..., help="Directory to sync."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Preview without modifying the index."),
):
    """Remove index entries for files that no longer exist on disk."""
    run_sync(directory, dry_run=dry_run)

if __name__ == "__main__":
    app()