import os, shutil, typer
from collections import defaultdict

from .db import get_connection

class UnionFind:
    def __init__(self):
        self._parent: dict[int, int] = {}

    def _ensure(self, x: int):
        if x not in self._parent:
            self._parent[x] = x

    def find(self, x: int) -> int:
        self._ensure(x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    def union(self, x: int, y: int):
        self._ensure(x)
        self._ensure(y)
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    def groups(self) -> list[set[int]]:
        buckets: dict[int, set[int]] = defaultdict(set)
        for x in self._parent:
            buckets[self.find(x)].add(x)
        return list(buckets.values())


def load_unresolved_matches(db_path: str) -> list[tuple[int, int]]:
    conn = get_connection(db_path)
    rows = conn.execute(
        "SELECT video_a_id, video_b_id FROM matches WHERE moved = 0"
    ).fetchall()
    return [(row["video_a_id"], row["video_b_id"]) for row in rows]


def load_video_paths(db_path: str, video_ids: list[int]) -> dict[int, str]:
    conn = get_connection(db_path)
    placeholders = ",".join("?" * len(video_ids))
    rows = conn.execute(
        f"SELECT id, path FROM videos WHERE id IN ({placeholders})", video_ids
    ).fetchall()
    return {row["id"]: row["path"] for row in rows}


def mark_group_moved(db_path: str, video_ids: list[int]):
    conn = get_connection(db_path)
    placeholders = ",".join("?" * len(video_ids))
    with conn:
        conn.execute(
            f"""
            UPDATE matches SET moved = 1
            WHERE video_a_id IN ({placeholders})
              AND video_b_id IN ({placeholders})
            """,
            video_ids + video_ids,
        )


def update_video_path(db_path: str, video_id: int, new_path: str):
    conn = get_connection(db_path)
    with conn:
        conn.execute(
            "UPDATE videos SET path = ? WHERE id = ?",
            (new_path, video_id),
        )


def build_groups(matches: list[tuple[int, int]]) -> list[set[int]]:
    uf = UnionFind()
    for a, b in matches:
        uf.union(a, b)
    return uf.groups()


def run_move(directory: str, dry_run: bool = False):
    """Main entry point for the move subcommand."""
    directory = os.path.abspath(directory)
    db_path = os.path.join(directory, ".matcha", "index.db")

    if not os.path.exists(db_path):
        typer.echo("No index found. Run `matcha index` first.")
        raise SystemExit(1)

    matches = load_unresolved_matches(db_path)
    if not matches:
        typer.echo("No pending matches to move.")
        return

    groups = build_groups(matches)
    all_ids = [vid_id for group in groups for vid_id in group]
    path_map = load_video_paths(db_path, all_ids)
    duplicates_dir = os.path.join(directory, "duplicates")

    if dry_run:
        typer.echo(
            f"[dry-run] {len(all_ids)} video(s) would be moved "
            f"into {len(groups)} subdirectory/ies under duplicates/"
        )
        return

    os.makedirs(duplicates_dir, exist_ok=True)

    existing = [
        int(d) for d in os.listdir(duplicates_dir)
        if d.isdigit() and os.path.isdir(os.path.join(duplicates_dir, d))
    ]
    next_group_num = max(existing, default=0) + 1

    groups_moved = 0
    videos_moved = 0

    for group in groups:
        group_dir = os.path.join(duplicates_dir, str(next_group_num))
        os.makedirs(group_dir, exist_ok=True)

        for vid_id in group:
            src = path_map.get(vid_id)
            if not src or not os.path.exists(src):
                typer.echo(f"  [SKIP] Video {vid_id} not found: {src}", err=True)
                continue

            filename = os.path.basename(src)
            dst = os.path.join(group_dir, filename)

            if os.path.exists(dst):
                name, ext = os.path.splitext(filename)
                dst = os.path.join(group_dir, f"{name}_{vid_id}{ext}")

            shutil.move(src, dst)
            update_video_path(db_path, vid_id, dst)
            videos_moved += 1

        mark_group_moved(db_path, list(group))
        next_group_num += 1
        groups_moved += 1

    typer.echo(
        f"Moved {videos_moved} video(s) into {groups_moved} "
        f"subdirectory/ies under {os.path.relpath(duplicates_dir, directory)}/"
    )