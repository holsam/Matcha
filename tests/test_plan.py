'''
MATCHA: tests for plan.py (synthetic library, ffprobe and prompts stubbed)
'''

# Import external dependencies
import json, os, pytest

# Import internal matcha objects
from matcha import plan
from matcha.db import get_connection, init_schema
from matcha.plan import Info

# _info: a probed file, 1080p at a healthy bitrate unless overridden
def _info(vid: int = 1, **kw) -> Info:
    base = dict(id=vid, path=f'/lib/v{vid}.mp4', size=100, duration=10.0, width=1920, height=1080, bitrate=8_000_000, fps=30.0, codec='h264')
    return Info(**{**base, **kw})

# _hex: 64 bit hashes as the concatenated hex string frame_hashes stores
def _hex(*values: int) -> list[str]:
    return [f'{v:016x}' for v in values]

# _ZERO, _HALF, _ONES: hashes 32 and 64 bits apart, so any threshold under 32 tells them apart
_ZERO, _HALF, _ONES = 0x0, 0x00000000FFFFFFFF, 0xFFFFFFFFFFFFFFFF

# TestFormatting: clock and byte formatting
class TestFormatting:
    # test_clock: mm:ss under an hour, h:mm:ss over
    def test_clock(self):
        assert plan._clock(0) == '00:00'
        assert plan._clock(65) == '01:05'
        assert plan._clock(3661) == '1:01:01'

    # test_fmt: unit steps at 1024
    def test_fmt(self):
        assert plan._fmt(512) == '512.0 B'
        assert plan._fmt(1024) == '1.0 KB'
        assert plan._fmt(5 * 1024**3) == '5.0 GB'
        assert plan._fmt(2 * 1024**4) == '2.00 TB'

# TestQuality: the keep recommendation inputs
class TestQuality:
    # test_zero_pixels: unknown dimensions score nothing
    def test_zero_pixels(self):
        assert plan._quality(_info(width=0, height=0)) == 0.0

    # test_starved_1080p_loses_to_fed_720p: bitrate matters, not just resolution
    def test_starved_1080p_loses_to_fed_720p(self):
        starved = _info(1, bitrate=200_000)
        fed = _info(2, width=1280, height=720, bitrate=6_000_000)
        assert plan._quality(fed) > plan._quality(starved)

    # test_efficient_codec_beats_h264: same bitrate, av1 stretches further
    def test_efficient_codec_beats_h264(self):
        assert plan._quality(_info(codec='av1', bitrate=1_000_000)) > plan._quality(_info(codec='h264', bitrate=1_000_000))

    # test_unknown_codec_counts_as_h264: no KeyError, same score as h264
    def test_unknown_codec_counts_as_h264(self):
        assert plan._quality(_info(codec='mystery')) == plan._quality(_info(codec='h264'))

    # test_score_breaks_ties_on_size: equal quality, larger file wins
    def test_score_breaks_ties_on_size(self):
        assert plan._score(_info(size=200)) > plan._score(_info(size=100))

    # test_identical: ignores id and path, sees size and rounded duration
    def test_identical(self):
        assert plan._identical(_info(1)) == plan._identical(_info(2))
        assert plan._identical(_info(1)) != plan._identical(_info(2, size=101))
        assert plan._identical(_info(1, duration=10.01)) == plan._identical(_info(2, duration=10.04))

# TestBuildGroups: connected components of the match graph
class TestBuildGroups:
    # test_empty: no pairs, no groups
    def test_empty(self):
        assert plan._build_groups([]) == []

    # test_chain_merges: a-b and b-c form one group
    def test_chain_merges(self):
        assert plan._build_groups([(1, 2), (2, 3)]) == [{1, 2, 3}]

    # test_separate_components: disjoint pairs stay apart
    def test_separate_components(self):
        groups = plan._build_groups([(1, 2), (3, 4), (2, 5)])
        assert sorted(sorted(g) for g in groups) == [[1, 2, 5], [3, 4]]

# TestBar: one diagram row
class TestBar:
    # test_width_is_constant: every row lines up
    def test_width_is_constant(self):
        for start, end in ((0, 10), (2, 4), (9.9, 10), (0, 0)):
            assert len(plan._bar(start, end, 10, 20)) == 20

    # test_full_span: a whole-source clip fills the bar
    def test_full_span(self):
        assert plan._bar(0, 10, 10, 20) == '-' * 20

    # test_minimum_one_char: a tiny clip is still visible
    def test_minimum_one_char(self):
        assert plan._bar(5, 5.001, 10, 20).count('-') == 1

    # test_offset: a clip from the midpoint starts half way along
    def test_offset(self):
        assert plan._bar(5, 10, 10, 20) == ' ' * 10 + '-' * 10

# db: connection to a synthetic index with three videos and frame hashes
@pytest.fixture()
def db(tmp_path):
    (tmp_path / '.matcha').mkdir()
    db_path = str(tmp_path / '.matcha' / 'index.db')
    init_schema(db_path)
    conn = get_connection(db_path)
    with conn:
        for vid in (1, 2, 3):
            conn.execute('INSERT INTO videos (id, path, duration) VALUES (?, ?, 10)', (vid, str(tmp_path / f'v{vid}.mp4')))
    return {'dir': str(tmp_path), 'root': tmp_path, 'conn': conn}

# _add_hashes: store one hash per second for a video
def _add_hashes(db, vid: int, hashes: list[str]) -> None:
    with db['conn'] as conn:
        conn.executemany('INSERT INTO frame_hashes (video_id, timestamp, phash) VALUES (?, ?, ?)', [(vid, float(t), h) for t, h in enumerate(hashes)])

# TestLocate: where a clip sits inside its source
class TestLocate:
    # test_finds_offset: clip taken from the middle of the source
    def test_finds_offset(self, db):
        _add_hashes(db, 1, _hex(_ZERO, _ONES, _HALF, _ZERO))
        _add_hashes(db, 2, _hex(_ONES, _HALF))
        assert plan._locate(db['conn'], 2, 1, 10) == (1.0, 1.0)

    # test_partial_match: fraction counts matching frames only
    def test_partial_match(self, db):
        _add_hashes(db, 1, _hex(_ZERO, _ONES, _HALF, _ZERO))
        _add_hashes(db, 2, _hex(_ONES, _ZERO))
        start, frac = plan._locate(db['conn'], 2, 1, 10)
        assert frac == 0.5

    # test_threshold_tolerates_noise: a few flipped bits still match
    def test_threshold_tolerates_noise(self, db):
        _add_hashes(db, 1, _hex(_ZERO, _ONES))
        _add_hashes(db, 2, _hex(_ONES ^ 0b111))
        assert plan._locate(db['conn'], 2, 1, 10) == (1.0, 1.0)
        assert plan._locate(db['conn'], 2, 1, 2)[1] == 0.0

    # test_clip_longer_than_source: nothing to align
    def test_clip_longer_than_source(self, db):
        _add_hashes(db, 1, _hex(_ZERO))
        _add_hashes(db, 2, _hex(_ZERO, _ONES))
        assert plan._locate(db['conn'], 2, 1, 10) == (0.0, 0.0)

    # test_no_hashes: an empty clip is not a crash
    def test_no_hashes(self, db):
        _add_hashes(db, 1, _hex(_ZERO, _ONES))
        assert plan._locate(db['conn'], 2, 1, 10) == (0.0, 0.0)

# TestPlanFile: persistence of decisions
class TestPlanFile:
    # test_round_trip: saved groups load back keyed by their key
    def test_round_trip(self, tmp_path):
        path = str(tmp_path / 'plan.json')
        entries = {'1,2': {'key': '1,2', 'files': []}, '3,4': {'key': '3,4', 'files': []}}
        plan._save_plan(path, entries)
        assert plan._load_plan(path) == entries
        assert json.load(open(path))['version'] == 1

    # test_atomic: no temp file is left behind
    def test_atomic(self, tmp_path):
        path = str(tmp_path / 'plan.json')
        plan._save_plan(path, {})
        assert os.listdir(tmp_path) == ['plan.json']

    # test_missing_or_corrupt: both read as an empty plan
    def test_missing_or_corrupt(self, tmp_path):
        assert plan._load_plan(str(tmp_path / 'nope.json')) == {}
        bad = tmp_path / 'bad.json'
        bad.write_text('{not json')
        assert plan._load_plan(str(bad)) == {}
        wrong = tmp_path / 'wrong.json'
        wrong.write_text('{"groups": [{"no_key": 1}]}')
        assert plan._load_plan(str(wrong)) == {}

# _Answer: stand-in for a questionary prompt, whose ask() returns a fixed answer
class _Answer:
    def __init__(self, value):
        self.value = value

    def ask(self):
        return self.value

# lib: db with videos 1 and 2 matched at 0.9 and files on disk, ffprobe stubbed (1 is the better file)
@pytest.fixture()
def lib(db, monkeypatch):
    infos = {1: _info(1, size=300), 2: _info(2, size=200, bitrate=1_000_000)}
    for vid in (1, 2):
        path = db['root'] / f'v{vid}.mp4'
        path.write_text('x' * infos[vid].size)
        infos[vid] = _info(vid, path=str(path), size=infos[vid].size, bitrate=infos[vid].bitrate)
        _add_hashes(db, vid, _hex(_ZERO, _ONES))
    with db['conn'] as conn:
        conn.execute("INSERT INTO matches (video_a_id, video_b_id, match_type, confidence, found_at) VALUES (1, 2, 'duplicate', 0.9, 0)")
    monkeypatch.setattr(plan, '_probe', lambda vid, path, duration: infos.get(vid))
    monkeypatch.setattr(plan, '_ask_name', lambda path: os.path.basename(path))
    db['infos'] = infos
    db['plan_path'] = str(db['root'] / '.matcha' / 'review_plan.json')
    return db

# _ticks: stub the checkbox prompt, recording the choices offered; answer is called with the choices
def _ticks(monkeypatch, answer) -> list:
    offered = []

    def checkbox(message, choices, **kw):
        offered.append(choices)
        return _Answer(answer(choices))

    monkeypatch.setattr(plan.questionary, 'checkbox', checkbox)
    return offered

# _decisions: action per video id from the written plan
def _decisions(lib) -> dict[int, str]:
    group = json.load(open(lib['plan_path']))['groups'][0]
    return {f['id']: f['action'] for f in group['files']}

# TestRunPlan: end to end through the prompts
class TestRunPlan:
    # test_missing_index: a clear exit rather than a traceback
    def test_missing_index(self, tmp_path):
        with pytest.raises(SystemExit):
            plan.run_plan(str(tmp_path))

    # test_recommended_keep_is_ticked: the better file starts ticked, the other does not
    def test_recommended_keep_is_ticked(self, lib, monkeypatch):
        offered = _ticks(monkeypatch, lambda choices: [c.value for c in choices if c.checked])
        plan.run_plan(lib['dir'])
        assert {c.value: c.checked for c in offered[0]} == {lib['infos'][1].path: True, lib['infos'][2].path: False}
        assert _decisions(lib) == {1: 'keep', 2: 'delete'}

    # test_unticked_are_deleted: overriding the recommendation is honoured
    def test_unticked_are_deleted(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][2].path])
        plan.run_plan(lib['dir'])
        assert _decisions(lib) == {1: 'delete', 2: 'keep'}

    # test_keeping_both: both ticked, nothing to delete
    def test_keeping_both(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [c.value for c in choices])
        plan.run_plan(lib['dir'])
        assert _decisions(lib) == {1: 'keep', 2: 'keep'}

    # test_rename: a changed name becomes a rename into the same folder
    def test_rename(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        monkeypatch.setattr(plan, '_ask_name', lambda path: 'better.mp4')
        plan.run_plan(lib['dir'])
        entry = next(f for f in json.load(open(lib['plan_path']))['groups'][0]['files'] if f['id'] == 1)
        assert entry['action'] == 'rename'
        assert entry['new_path'] == os.path.join(lib['dir'], 'better.mp4')

    # test_plan_records_sizes_and_key: what apply needs to count freed bytes and match the group
    def test_plan_records_sizes_and_key(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        plan.run_plan(lib['dir'])
        group = json.load(open(lib['plan_path']))['groups'][0]
        assert group['key'] == '1,2'
        assert {f['id']: f['size'] for f in group['files']} == {1: 300, 2: 200}

    # test_resume_skips_decided_groups: a second run asks nothing
    def test_resume_skips_decided_groups(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        plan.run_plan(lib['dir'])
        offered = _ticks(monkeypatch, lambda choices: pytest.fail('asked again'))
        plan.run_plan(lib['dir'])
        assert offered == []

    # test_restart_asks_again: --restart ignores the saved decisions
    def test_restart_asks_again(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        plan.run_plan(lib['dir'])
        offered = _ticks(monkeypatch, lambda choices: [lib['infos'][2].path])
        plan.run_plan(lib['dir'], restart=True)
        assert len(offered) == 1
        assert _decisions(lib) == {1: 'delete', 2: 'keep'}

    # test_ctrl_c_finishes_without_saving: a cancelled group is not recorded
    def test_ctrl_c_finishes_without_saving(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: None)
        plan.run_plan(lib['dir'])
        assert not os.path.exists(lib['plan_path'])

    # test_cancelled_rename_finishes_without_saving: Ctrl-C at the name prompt drops the whole group
    def test_cancelled_rename_finishes_without_saving(self, lib, monkeypatch):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        monkeypatch.setattr(plan, '_ask_name', lambda path: None)
        plan.run_plan(lib['dir'])
        assert not os.path.exists(lib['plan_path'])

    # test_no_pause_never_prompts_or_writes: a read-only listing
    def test_no_pause_never_prompts_or_writes(self, lib, monkeypatch):
        offered = _ticks(monkeypatch, lambda choices: pytest.fail('prompted'))
        plan.run_plan(lib['dir'], no_pause=True)
        assert offered == []
        assert not os.path.exists(lib['plan_path'])

    # test_custom_plan_path: --plan decides where decisions go
    def test_custom_plan_path(self, lib, monkeypatch, tmp_path_factory):
        _ticks(monkeypatch, lambda choices: [lib['infos'][1].path])
        custom = str(tmp_path_factory.mktemp('elsewhere') / 'mine.json')
        plan.run_plan(lib['dir'], plan=custom)
        assert os.path.exists(custom)
        assert not os.path.exists(lib['plan_path'])

    # test_low_confidence_not_grouped: below min_confidence nothing is offered
    def test_low_confidence_not_grouped(self, lib, monkeypatch):
        offered = _ticks(monkeypatch, lambda choices: pytest.fail('prompted'))
        plan.run_plan(lib['dir'], min_confidence=0.95)
        assert offered == []

    # test_dismissed_not_grouped: dismissed matches are left alone
    def test_dismissed_not_grouped(self, lib, monkeypatch):
        with lib['conn'] as conn:
            conn.execute('UPDATE matches SET dismissed = 1')
        offered = _ticks(monkeypatch, lambda choices: pytest.fail('prompted'))
        plan.run_plan(lib['dir'])
        assert offered == []

    # test_missing_file_leaves_nothing_to_compare: one file on disk, no prompt
    def test_missing_file_leaves_nothing_to_compare(self, lib, monkeypatch):
        monkeypatch.setattr(plan, '_probe', lambda vid, path, duration: lib['infos'][1] if vid == 1 else None)
        offered = _ticks(monkeypatch, lambda choices: pytest.fail('prompted'))
        plan.run_plan(lib['dir'])
        assert offered == []

    # test_identical_files_have_no_preticked_keep: a tie leaves the choice to the user
    def test_identical_files_have_no_preticked_keep(self, lib, monkeypatch):
        twin = lib['infos'][1]
        monkeypatch.setattr(plan, '_probe', lambda vid, path, duration: _info(vid, path=lib['infos'][vid].path, size=twin.size))
        offered = _ticks(monkeypatch, lambda choices: [choices[0].value])
        plan.run_plan(lib['dir'])
        assert not any(c.checked for c in offered[0])

    # test_subclip_is_offered_but_not_a_keep_candidate: a short member never beats the full-length file
    def test_subclip_is_offered_but_not_a_keep_candidate(self, lib, monkeypatch):
        lib['infos'][2] = _info(2, path=lib['infos'][2].path, size=999, duration=2.0, bitrate=20_000_000)
        offered = _ticks(monkeypatch, lambda choices: [c.value for c in choices if c.checked])
        plan.run_plan(lib['dir'])
        assert {c.value: c.checked for c in offered[0]}[lib['infos'][1].path] is True
