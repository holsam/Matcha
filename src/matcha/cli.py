import os, subprocess, sys
import typer

from matcha.apply import run as run_apply
from matcha.borders import Params, run_list, run_scan
from matcha.continuer import run_continue
from matcha.db import fold_in_moved_paths, init_schema
from matcha.plan import run_plan
from matcha.indexer import run_index
from matcha.matcher import run_match
from matcha.reencode import run_reencode

app = typer.Typer(help="Matcha — a video matching tool using perceptual hashing.")
borders_app = typer.Typer(help="Find videos with black borders and plan crops.")
app.add_typer(borders_app, name="borders", rich_help_panel="Additional utilities")
_D = Params()  # defaults for the borders options


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

@app.command(rich_help_panel="Review matches")
def plan(
    directory: str = typer.Argument(..., help="Directory containing .matcha/index.db."),
    min_confidence: float = typer.Option(0.8, help="Minimum match confidence to group on."),
    threshold: int = typer.Option(10, help="Max Hamming distance per frame, as used by `matcha match`."),
    no_pause: bool = typer.Option(False, "--no-pause", help="Print every group without waiting for input."),
    plan: str | None = typer.Option(None, "--plan", help="Plan file to write and resume from."),
    restart: bool = typer.Option(False, "--restart", help="Ignore any existing plan and start again."),
):
    """Step through duplicate groups in the terminal and write a review plan."""
    run_plan(directory, min_confidence=min_confidence, threshold=threshold, no_pause=no_pause, plan=plan, restart=restart)

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

@app.command(rich_help_panel="Additional utilities")
def reencode(
    directory: str = typer.Argument(..., help="Directory of videos to profile or re-encode."),
    bpp: float = typer.Option(0.06, help="Target bits per pixel per frame (HEVC): ~0.04 aggressive, 0.06 balanced, 0.10 high quality."),
    min_saving_pct: float = typer.Option(30.0, help="Minimum estimated saving per file, in percent."),
    min_saving_mb: float = typer.Option(50.0, help="Minimum estimated saving per file, in MB."),
    probe_workers: int = typer.Option(4, help="Number of parallel ffprobe workers."),
    preview: int = typer.Option(0, help="Build side-by-side comparison clips for N random candidates and stop."),
    bpps: str = typer.Option("0.03,0.06,0.10", help="Comma separated bpp values to compare with --preview."),
    preview_dir: str | None = typer.Option(None, "--preview-dir", help="Where --preview writes (default: .matcha/tmp)."),
    execute: bool = typer.Option(False, "--apply", help="Re-encode the candidates in place (default is a dry run)."),
    preset: str = typer.Option("medium", help="x265 preset, ultrafast to veryslow. Faster presets give larger files."),
    yes: bool = typer.Option(False, "--yes", help="Skip the confirmation prompt."),
    keep_originals: str | None = typer.Option(None, "--keep-originals", help="Move originals here, with a symlink to each new file, instead of deleting."),
):
    """Profile videos by codec and bitrate, and re-encode bloated ones to HEVC."""
    run_reencode(
        directory,
        bpp=bpp,
        min_saving_pct=min_saving_pct,
        min_saving_mb=min_saving_mb,
        probe_workers=probe_workers,
        preview=preview,
        bpps=bpps,
        preview_dir=preview_dir,
        apply=execute,
        preset=preset,
        yes=yes,
        keep_originals=keep_originals,
    )

@borders_app.command()
def scan(
    directory: str = typer.Argument(..., help="Directory to scan."),
    workers: int = typer.Option(4, help="Number of parallel scan workers."),
    rescan: bool = typer.Option(False, "--rescan", help="Scan every video again, even if up to date."),
    prune: bool = typer.Option(False, "--prune", help="Forget stored videos that no longer exist."),
    flush_every: float = typer.Option(30.0, help="Seconds between saves of .matcha/borders.json."),
    black_tol: int = typer.Option(_D.black_tol, help="Luma above the frame's own black level still counted as black."),
    max_black_ref: int = typer.Option(_D.max_black_ref, help="Cap on the black level estimated from a frame."),
    blank_frac: float = typer.Option(_D.blank_frac, help="Share of black pixels that makes a row or column blank."),
    min_margin_frac: float = typer.Option(_D.min_margin_frac, help="Ignore margins under this share of the frame."),
    min_content_luma: float = typer.Option(_D.min_content_luma, help="Mean luma the picture needs for a frame to count."),
    min_edge_step: float = typer.Option(_D.min_edge_step, help="Luma step needed across a border edge."),
    edge_band: int = typer.Option(_D.edge_band, help="Lines each side of an edge used for the step."),
    agree_px: int = typer.Option(_D.agree_px, help="Margins agree when within this many pixels."),
    agree_frac: float = typer.Option(_D.agree_frac, help="Share of informative frames that must agree."),
    min_informative_a: int = typer.Option(_D.min_informative_a, help="Informative frames needed to finish after the quick pass."),
    min_informative_b: int = typer.Option(_D.min_informative_b, help="Informative frames needed after the full pass."),
    frames_a: int = typer.Option(_D.frames_a, help="Frames in the quick pass."),
    frames_b: int = typer.Option(_D.frames_b, help="Frames in the full pass."),
    span_start: float = typer.Option(_D.span[0], help="Start of the sampled part of each video (0-1)."),
    span_end: float = typer.Option(_D.span[1], help="End of the sampled part of each video (0-1)."),
    max_width: int = typer.Option(_D.max_width, help="Frames are scaled down to this width before analysis."),
):
    """Scan videos in DIRECTORY for black borders and record them in .matcha/borders.json."""
    if not os.path.isdir(directory):
        typer.echo(f"Not a directory: {directory}")
        raise SystemExit(1)
    params = Params(
        black_tol=black_tol, max_black_ref=max_black_ref, blank_frac=blank_frac, min_margin_frac=min_margin_frac,
        min_content_luma=min_content_luma, min_edge_step=min_edge_step, edge_band=edge_band, agree_px=agree_px,
        agree_frac=agree_frac, min_informative_a=min_informative_a, min_informative_b=min_informative_b,
        frames_a=frames_a, frames_b=frames_b, span=(span_start, span_end), max_width=max_width,
    )
    run_scan(directory, params=params, workers=workers, rescan=rescan, flush_every=flush_every, prune=prune)

@borders_app.command(name="list")
def list_(
    directory: str = typer.Argument(..., help="Directory containing .matcha/borders.json."),
    include_variable: bool = typer.Option(False, "--include-variable", help="Also list videos whose borders vary, for review."),
    min_confidence: float = typer.Option(0.0, help="Minimum share of frames that agreed (0.0-1.0)."),
    sort: str = typer.Option("path", help="Sort by path, confidence or area (pixels removed)."),
    as_json: bool = typer.Option(False, "--json", help="Print machine-readable JSON instead of a table."),
):
    """List videos with detected borders and their crop boxes."""
    raise SystemExit(run_list(directory, include_variable=include_variable, min_confidence=min_confidence, sort=sort, as_json=as_json))

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