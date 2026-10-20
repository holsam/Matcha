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
