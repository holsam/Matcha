'''
MATCHA: derive duplicate groups from recorded matches.
'''

# Import external dependencies
from collections import defaultdict
from dataclasses import dataclass

# Import internal matcha objects
from matcha.db import get_connection


# UnionFind: disjoint-set over video ids, used to merge pairwise matches into groups
class UnionFind:
    # __init__: start with no elements
    def __init__(self) -> None:
        self._parent: dict[int, int] = {}

    # find: root of x, with path compression
    def find(self, x: int) -> int:
        self._parent.setdefault(x, x)
        if self._parent[x] != x:
            self._parent[x] = self.find(self._parent[x])
        return self._parent[x]

    # union: merge the sets containing x and y
    def union(self, x: int, y: int) -> None:
        rx, ry = self.find(x), self.find(y)
        if rx != ry:
            self._parent[ry] = rx

    # groups: every set as a set of ids
    def groups(self) -> list[set[int]]:
        buckets: dict[int, set[int]] = defaultdict(set)
        for x in list(self._parent):
            buckets[self.find(x)].add(x)
        return list(buckets.values())


# build_groups: connected components of the match graph
def build_groups(matches: list[tuple[int, int]]) -> list[set[int]]:
    uf = UnionFind()
    for a, b in matches:
        uf.union(a, b)
    return uf.groups()


# Group: summary of one duplicate group, built from matches and durations only (no file access)
@dataclass(frozen=True)
class Group:
    key: int  # smallest video id, stable while membership is unchanged
    video_ids: tuple[int, ...]
    max_confidence: float
    has_subclip: bool
    total_duration: float


# find_groups: undismissed duplicate groups, largest first then most confident
def find_groups(db_path: str) -> list[Group]:
    conn = get_connection(db_path)
    rows = conn.execute(
        '''
        SELECT m.video_a_id AS a, m.video_b_id AS b, m.match_type, m.confidence,
               COALESCE(va.duration, 0) AS dur_a, COALESCE(vb.duration, 0) AS dur_b
        FROM matches m
        JOIN videos va ON va.id = m.video_a_id
        JOIN videos vb ON vb.id = m.video_b_id
        WHERE m.dismissed = 0
        '''
    ).fetchall()

    members = build_groups([(r['a'], r['b']) for r in rows])
    owner = {vid: i for i, group in enumerate(members) for vid in group}
    confidence = [0.0] * len(members)
    subclip = [False] * len(members)
    durations: list[dict[int, float]] = [{} for _ in members]
    for r in rows:
        i = owner[r['a']]
        confidence[i] = max(confidence[i], r['confidence'])
        subclip[i] = subclip[i] or r['match_type'] == 'subclip'
        durations[i][r['a']] = r['dur_a']
        durations[i][r['b']] = r['dur_b']

    groups = [
        Group(min(ids), tuple(sorted(ids)), confidence[i], subclip[i], sum(durations[i].values()))
        for i, ids in enumerate(members)
    ]
    groups.sort(key=lambda g: (-len(g.video_ids), -g.max_confidence, g.key))
    return groups


# SORTS: sort key name to (label, key function, reverse)
SORTS = {
    'size': ('Most videos', lambda g: (len(g.video_ids), g.max_confidence), True),
    'confidence': ('Highest confidence', lambda g: (g.max_confidence, len(g.video_ids)), True),
    'duration': ('Longest total duration', lambda g: g.total_duration, True),
}


# sort_groups: reorder groups by one of the SORTS keys, ties broken by group key for a stable order
def sort_groups(groups: list[Group], sort: str) -> list[Group]:
    _, key, reverse = SORTS[sort]
    return sorted(sorted(groups, key=lambda g: g.key), key=key, reverse=reverse)


# paginate: one page of items plus the page count (always at least 1), page clamped into range
def paginate[T](items: list[T], page: int, per_page: int) -> tuple[list[T], int, int]:
    pages = max(1, -(-len(items) // per_page))
    page = min(max(page, 0), pages - 1)
    return items[page * per_page:(page + 1) * per_page], page, pages


# video_names: file names for the given ids, used to label group cards without touching the disk
def video_names(db_path: str, video_ids: list[int]) -> dict[int, str]:
    conn = get_connection(db_path)
    marks = ','.join('?' * len(video_ids))
    rows = conn.execute(f'SELECT id, path FROM videos WHERE id IN ({marks})', video_ids).fetchall()
    return {r['id']: r['path'].rsplit('/', 1)[-1] for r in rows}


# matches_version: cheap token that changes whenever the set of visible matches changes, for cache keys
def matches_version(db_path: str) -> tuple[int, int, int]:
    conn = get_connection(db_path)
    row = conn.execute(
        'SELECT COALESCE(MAX(id), 0), COUNT(*), COALESCE(SUM(dismissed), 0) FROM matches'
    ).fetchone()
    return (row[0], row[1], row[2])


# resolve_group: the current group that best overlaps the ids a user had open, or None once they are all gone
def resolve_group(groups: list[Group], video_ids) -> Group | None:
    wanted = set(video_ids)
    best = max(groups, key=lambda g: len(wanted & set(g.video_ids)), default=None)
    return best if best and wanted & set(best.video_ids) else None


# VideoRow: the stored facts about one video
@dataclass(frozen=True)
class VideoRow:
    id: int
    path: str
    duration: float


# load_videos: stored rows for the given ids, in id order
def load_videos(db_path: str, video_ids: list[int]) -> list[VideoRow]:
    conn = get_connection(db_path)
    marks = ','.join('?' * len(video_ids))
    rows = conn.execute(
        f'SELECT id, path, COALESCE(duration, 0) AS duration FROM videos WHERE id IN ({marks}) ORDER BY id',
        list(video_ids),
    ).fetchall()
    return [VideoRow(r['id'], r['path'], r['duration']) for r in rows]


# Pair: one recorded match between two videos
@dataclass(frozen=True)
class Pair:
    a: int
    b: int
    match_type: str
    confidence: float


# load_pairs: undismissed matches whose two videos are both in the given ids, most similar first
def load_pairs(db_path: str, video_ids: list[int]) -> list[Pair]:
    conn = get_connection(db_path)
    marks = ','.join('?' * len(video_ids))
    rows = conn.execute(
        f'''
        SELECT video_a_id, video_b_id, match_type, confidence FROM matches
        WHERE dismissed = 0 AND video_a_id IN ({marks}) AND video_b_id IN ({marks})
        ORDER BY confidence DESC, id
        ''',
        [*video_ids, *video_ids],
    ).fetchall()
    return [Pair(r['video_a_id'], r['video_b_id'], r['match_type'], r['confidence']) for r in rows]
