'''
MATCHA: tests for apply.py (synthetic library)
'''

# Import external dependencies
import io, json, os, pathlib, pytest
from rich.console import Console

# Import internal matcha objects
from matcha import apply
from matcha.db import get_connection, init_schema

# lib: library of three videos (1 and 2 duplicates, 3 a subclip of both) with rows in every table
@pytest.fixture()
def lib(tmp_path):
    (tmp_path / '.matcha').mkdir()
    db_path = str(tmp_path / '.matcha' / 'index.db')
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        for vid in (1, 2, 3):
            f = tmp_path / f'v{vid}.mp4'
            f.write_text('x' * vid)
            conn.execute('INSERT INTO videos (id, path, duration) VALUES (?, ?, 1)', (vid, str(f)))
            conn.execute("INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, 0, 'ff')", (vid,))
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (1, 2, 'duplicate', 0.9, 0)")
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (2, 3, 'subclip', 0.9, 0)")
        conn.execute('INSERT INTO candidate_pairs (video_a_id, video_b_id) VALUES (1, 3)')
        conn.execute('INSERT INTO candidate_search_progress (video_id, queried_at) VALUES (1, 0)')
    return {'dir': str(tmp_path), 'root': tmp_path, 'conn': conn}

# _plan: write a plan with one group; actions maps video id to 'keep' / 'delete' / a new file name for a rename
def _plan(lib, actions: dict[int, str]) -> str:
    files = []
    for vid, action in actions.items():
        path = str(lib['root'] / f'v{vid}.mp4')
        entry = {'id': vid, 'path': path, 'size': vid, 'action': 'keep' if action == 'keep' else 'delete' if action == 'delete' else 'rename'}
        if entry['action'] == 'rename':
            entry['new_path'] = str(lib['root'] / action)
        files.append(entry)
    plan_path = str(lib['root'] / '.matcha' / 'review_plan.json')
    with open(plan_path, 'w') as f:
        json.dump({'version': 1, 'groups': [{'key': '1,2,3', 'files': files}]}, f)
    return plan_path

# _apply: run the script quietly, returns the exit code
def _apply(lib, **kwargs) -> int:
    return apply.run(lib['dir'], apply=True, yes=True, quiet=True, console=Console(file=io.StringIO()), **kwargs)

# _count: rows in a table
def _count(lib, table: str) -> int:
    return lib['conn'].execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0]

# TestDryRun: the default changes nothing
class TestDryRun:
    # test_nothing_changes: files, rows and plan are untouched
    def test_nothing_changes(self, lib):
        plan_path = _plan(lib, {1: 'keep', 2: 'delete', 3: 'renamed.mp4'})
        before = open(plan_path).read()
        code = apply.run(lib['dir'], quiet=True, console=Console(file=io.StringIO()))
        assert code == 0
        assert (lib['root'] / 'v2.mp4').exists() and (lib['root'] / 'v3.mp4').exists()
        assert _count(lib, 'videos') == 3
        assert open(plan_path).read() == before

# TestDelete: file removal then row purge
class TestDelete:
    # test_removes_file_and_all_rows: no orphan rows under foreign_keys=ON
    def test_removes_file_and_all_rows(self, lib):
        _plan(lib, {2: 'keep', 1: 'delete'})
        assert _apply(lib) == 0
        assert not (lib['root'] / 'v1.mp4').exists()
        assert _count(lib, 'videos') == 2
        assert _count(lib, 'frame_hashes') == 2
        assert _count(lib, 'candidate_pairs') == 0
        assert _count(lib, 'candidate_search_progress') == 0
        assert _count(lib, 'matches') == 1  # only the 2-3 match survives
        assert lib['conn'].execute('PRAGMA foreign_key_check').fetchall() == []

    # test_missing_file_still_purged: an already-gone file does not block the row purge
    def test_missing_file_still_purged(self, lib):
        _plan(lib, {2: 'keep', 1: 'delete'})
        (lib['root'] / 'v1.mp4').unlink()
        assert _apply(lib) == 0
        assert lib['conn'].execute('SELECT COUNT(*) FROM videos WHERE id = 1').fetchone()[0] == 0

    # test_remove_failure_keeps_rows: a failed file step leaves the db untouched and reports a failure
    def test_remove_failure_keeps_rows(self, lib, monkeypatch):
        _plan(lib, {2: 'keep', 1: 'delete'})

        def boom(path):
            raise PermissionError('denied')

        monkeypatch.setattr(os, 'remove', boom)
        assert _apply(lib) == 1
        assert (lib['root'] / 'v1.mp4').exists()
        assert lib['conn'].execute('SELECT COUNT(*) FROM videos WHERE id = 1').fetchone()[0] == 1

    # test_last_copy_guard: nothing is deleted when no survivor exists on disk
    def test_last_copy_guard(self, lib):
        _plan(lib, {1: 'delete', 2: 'delete'})
        assert _apply(lib) == 0
        assert (lib['root'] / 'v1.mp4').exists() and (lib['root'] / 'v2.mp4').exists()
        assert _count(lib, 'videos') == 3

    # test_survivor_missing_on_disk: a keep whose file is gone does not license deleting the others
    def test_survivor_missing_on_disk(self, lib):
        _plan(lib, {2: 'keep', 1: 'delete'})
        (lib['root'] / 'v2.mp4').unlink()
        assert _apply(lib) == 0
        assert (lib['root'] / 'v1.mp4').exists()

    # test_stale_plan_path_mismatch: a row whose path differs from the plan is left alone
    def test_stale_plan_path_mismatch(self, lib):
        _plan(lib, {2: 'keep', 1: 'delete'})
        with lib['conn']:
            lib['conn'].execute("UPDATE videos SET path = '/elsewhere/v1.mp4' WHERE id = 1")
        assert _apply(lib) == 0
        assert (lib['root'] / 'v1.mp4').exists()
        assert _count(lib, 'videos') == 3

# TestRename: file rename and path update
class TestRename:
    # test_renames_file_and_row: disk and videos.path agree afterwards
    def test_renames_file_and_row(self, lib):
        _plan(lib, {1: 'keep', 3: 'best.mp4'})
        assert _apply(lib) == 0
        assert (lib['root'] / 'best.mp4').exists() and not (lib['root'] / 'v3.mp4').exists()
        assert lib['conn'].execute('SELECT path FROM videos WHERE id = 3').fetchone()[0] == str(lib['root'] / 'best.mp4')

    # test_collision_fails: an existing target is reported and nothing moves
    def test_collision_fails(self, lib):
        _plan(lib, {1: 'keep', 3: 'v2.mp4'})
        assert _apply(lib) == 1
        assert (lib['root'] / 'v3.mp4').exists()
        assert lib['conn'].execute('SELECT path FROM videos WHERE id = 3').fetchone()[0] == str(lib['root'] / 'v3.mp4')

    # test_db_failure_undoes_rename: a failed row update puts the file back
    def test_db_failure_undoes_rename(self, lib, monkeypatch):
        _plan(lib, {1: 'keep', 3: 'best.mp4'})
        monkeypatch.setattr(apply, '_update_path', lambda conn, vid, dst: ('failed', 'locked'))
        assert _apply(lib) == 1
        assert (lib['root'] / 'v3.mp4').exists() and not (lib['root'] / 'best.mp4').exists()

    # test_freed_name_reused: deleting a file frees its name for a rename in the same group
    def test_freed_name_reused(self, lib):
        _plan(lib, {1: 'keep', 2: 'delete', 3: 'v2.mp4'})
        assert _apply(lib) == 0
        assert (lib['root'] / 'v2.mp4').read_text() == 'xxx'

# TestDismiss: matches between survivors
class TestDismiss:
    # test_survivors_dismissed: a kept pair no longer shows as an undecided match
    def test_survivors_dismissed(self, lib):
        _plan(lib, {1: 'keep', 2: 'keep', 3: 'delete'})
        assert _apply(lib) == 0
        assert lib['conn'].execute('SELECT dismissed FROM matches WHERE video_a_id = 1 AND video_b_id = 2').fetchone()[0] == 1

    # test_single_survivor_no_dismiss: nothing to dismiss with one survivor
    def test_single_survivor_no_dismiss(self, lib):
        _plan(lib, {1: 'keep', 2: 'delete'})
        assert _apply(lib) == 0
        assert lib['conn'].execute('SELECT COUNT(*) FROM matches WHERE dismissed = 1').fetchone()[0] == 0

# TestResume: the plan records progress
class TestResume:
    # test_second_run_does_nothing: applied entries are stamped and skipped
    def test_second_run_does_nothing(self, lib):
        plan_path = _plan(lib, {1: 'keep', 2: 'delete', 3: 'best.mp4'})
        assert _apply(lib) == 0
        stamped = json.load(open(plan_path))['groups'][0]['files']
        assert all(f['applied']['db'].startswith('ok') for f in stamped if f['action'] != 'keep')
        out = io.StringIO()
        assert apply.run(lib['dir'], apply=True, yes=True, quiet=True, console=Console(file=out)) == 0
        assert 'nothing to apply' in out.getvalue()

    # test_db_step_resumes_after_file_step: a crash between the file and db step finishes only the db step
    def test_db_step_resumes_after_file_step(self, lib):
        plan_path = _plan(lib, {1: 'keep', 2: 'delete'})
        (lib['root'] / 'v2.mp4').unlink()
        plan = json.load(open(plan_path))
        plan['groups'][0]['files'][1]['applied'] = {'file': 'ok: removed'}
        json.dump(plan, open(plan_path, 'w'))
        assert _apply(lib) == 0
        assert lib['conn'].execute('SELECT COUNT(*) FROM videos WHERE id = 2').fetchone()[0] == 0

# _vectors: stub FAISS vector removal, returning the list of (index dir, ids) calls
@pytest.fixture()
def vectors(monkeypatch):
    calls = []
    monkeypatch.setattr('matcha.faiss_index.remove_videos_from_index', lambda d, ids: calls.append((d, sorted(ids))) or len(calls[-1][1]))
    return calls

# TestVectors: deleted videos leave the FAISS index too
class TestVectors:
    # test_deleted_ids_removed_once: one pass for the whole plan, pointed at .matcha
    def test_deleted_ids_removed_once(self, lib, vectors):
        _plan(lib, {1: 'keep', 2: 'delete', 3: 'delete'})
        assert _apply(lib) == 0
        assert vectors == [(str(lib['root'] / '.matcha'), [2, 3])]

    # test_recorded_in_plan: the slot stops a second run repeating it
    def test_recorded_in_plan(self, lib, vectors):
        plan_path = _plan(lib, {1: 'keep', 2: 'delete'})
        _apply(lib)
        applied = json.load(open(plan_path))['groups'][0]['files'][1]['applied']
        assert applied['index'] == 'ok'
        assert _apply(lib) == 0
        assert len(vectors) == 1

    # test_renames_and_keeps_untouched: nothing to remove, faiss never loaded
    def test_renames_and_keeps_untouched(self, lib, vectors):
        _plan(lib, {1: 'keep', 2: 'renamed.mp4'})
        _apply(lib)
        assert vectors == []

    # test_dry_run_leaves_index_alone: no removal when nothing changes
    def test_dry_run_leaves_index_alone(self, lib, vectors):
        _plan(lib, {1: 'keep', 2: 'delete'})
        apply.run(lib['dir'], quiet=True, console=Console(file=io.StringIO()))
        assert vectors == []

    # test_failure_reported_and_retried: the plan stays unmarked so the next run tries again
    def test_failure_reported_and_retried(self, lib, monkeypatch, vectors):
        def boom(d, ids):
            raise RuntimeError('index locked')
        monkeypatch.setattr('matcha.faiss_index.remove_videos_from_index', boom)
        plan_path = _plan(lib, {1: 'keep', 2: 'delete'})
        assert _apply(lib) == 1
        assert 'index' not in json.load(open(plan_path))['groups'][0]['files'][1]['applied']
        monkeypatch.setattr('matcha.faiss_index.remove_videos_from_index', lambda d, ids: vectors.append(ids) or 1)
        assert _apply(lib) == 0
        assert vectors == [[2]]

    # test_plan_applied_before_vectors_existed: rows already purged, vectors still owed
    def test_plan_applied_before_vectors_existed(self, lib, vectors):
        plan_path = _plan(lib, {1: 'keep', 2: 'delete'})
        assert _apply(lib) == 0
        plan = json.load(open(plan_path))
        del plan['groups'][0]['files'][1]['applied']['index']
        json.dump(plan, open(plan_path, 'w'))
        vectors.clear()
        assert _apply(lib) == 0
        assert vectors == [(str(lib['root'] / '.matcha'), [2])]

# TestDbHelpers: the shared purge and dismiss helpers
class TestDbHelpers:
    # test_purge_video_counts_rows: every referencing row goes and is counted
    def test_purge_video_counts_rows(self, lib):
        from matcha.db import purge_video
        # video 3: videos row, 1 frame hash, 1 match (2,3), 1 candidate pair (1,3)
        assert purge_video(lib['conn'], 3) == 4
        assert _count(lib, 'videos') == 2

    # test_dismiss_matches_counts: only matches inside the given set
    def test_dismiss_matches_counts(self, lib):
        from matcha.db import dismiss_matches
        assert dismiss_matches(lib['conn'], [1, 2]) == 1
