import click
from .indexer import run_index


@click.group()
def cli():
    """Matcha — a video matching tool using perceptual hashing."""
    pass


@cli.command()
@click.argument("directory", type=click.Path(exists=True, file_okay=False))
@click.option("--fps", default=1.0, show_default=True, help="Frames per second to sample for hashing.")
@click.option("--workers", default=4, show_default=True, help="Number of parallel indexing workers.")
def index(directory, fps, workers):
    """Fingerprint all videos in DIRECTORY and store results in .matcha/index.db."""
    run_index(directory, fps=fps, workers=workers)