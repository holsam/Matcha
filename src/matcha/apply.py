'''
MATCHA: apply a review plan, making the file changes and keeping the index db consistent; deletes are permanent (os.remove). Stop `matcha index/match/review` while applying.
'''

# Import external dependencies
import json, os, sqlite3, time
from collections import Counter
from contextlib import nullcontext
from dataclasses import dataclass
from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TaskProgressColumn, TextColumn, TimeElapsedColumn
from rich.spinner import Spinner
from rich.table import Table
from rich.text import Text

# _VIDEO_TABLES: tables keyed by a single video_id, purged before the video row (FK order)
_VIDEO_TABLES = ('frame_hashes', 'audio_fingerprints', 'candidate_search_progress')

# _PAIR_TABLES: tables keyed by a pair of video ids, purged before the video row (FK order)
_PAIR_TABLES = ('matches', 'comparisons', 'candidate_pairs')

# _ICONS: status to (glyph, style) for finished steps
_ICONS = {
    'ok': ('✔', 'green'), 'failed': ('✘', 'red'), 'skipped': ('–', 'dim'),
    'would': ('○', 'cyan'), 'pending': ('·', 'dim'), 'info': ('·', 'dim'),
}

# _fmt: human readable byte count
def _fmt(n: float) -> str:
    for unit in ('B', 'KB', 'MB', 'GB'):
        if n < 1024:
            return f'{n:.1f} {unit}'
        n /= 1024
    return f'{n:.2f} TB'

# Step: one operation shown as a row, either on the filesystem or on the db
@dataclass
class Step:
    kind: str  # 'file' | 'db' | 'info'
    verb: str
    text: str
    status: str = 'pending'  # pending | running | ok | failed | skipped | would | info
    detail: str = ''

    # label: left column text, e.g. 'delete file'
    @property
    def label(self) -> str:
        return self.verb if self.kind == 'info' else f'{self.verb} {self.kind}'

# Display: live panel for the current group plus an overall progress bar, or plain lines when not a terminal
class Display:
    # __init__: live only on a terminal and when not quiet
    def __init__(self, console: Console, *, quiet: bool, total: int) -> None:
        self.console = console
        self.quiet = quiet
        self.steps: list[Step] = []
        self.title = ''
        self.live_on = console.is_terminal and not quiet
        self.progress = Progress(
            SpinnerColumn(), BarColumn(), TaskProgressColumn(), TextColumn('{task.completed}/{task.total} groups'),
            TimeElapsedColumn(), TextColumn('[green]{task.fields[freed]}[/] freed'), console=console,
        )
        self.task = self.progress.add_task('groups', total=total, freed=_fmt(0))
        self.live = Live(get_renderable=self._layout, console=console, refresh_per_second=12) if self.live_on else None

    # __enter__: start the live display when enabled
    def __enter__(self) -> 'Display':
        if self.live:
            self.live.__enter__()
        return self

    # __exit__: stop the live display
    def __exit__(self, *exc) -> None:
        if self.live:
            self.live.__exit__(*exc)

    # _panel: the current group as a table of steps, spinner while running
    def _panel(self) -> Panel:
        grid = Table.grid(padding=(0, 1))
        for _ in range(4):
            grid.add_column(no_wrap=True, overflow='ellipsis')
        for s in self.steps:
            if s.status == 'running':
                icon = Spinner('dots', style='yellow')
            else:
                glyph, style = _ICONS[s.status]
                icon = Text(glyph, style=style)
            label = Text(('would ' if s.status == 'would' else '') + s.label, style='dim' if s.status in ('skipped', 'info') else '')
            grid.add_row(icon, label, Text(s.text), Text(s.detail, style='red' if s.status == 'failed' else 'dim'))
        return Panel(grid, title=self.title, title_align='left')

    # _layout: panel above the overall bar, bar only between groups
    def _layout(self) -> Group:
        return Group(self._panel(), self.progress) if self.steps else Group(self.progress)

    # refresh: redraw after a step changed
    def refresh(self) -> None:
        if self.live:
            self.live.refresh()

    # start_group: begin a new group block
    def start_group(self, title: str, steps: list[Step]) -> None:
        self.title, self.steps = title, steps
        if not self.live_on and not self.quiet:
            self.console.print(f'[bold]{title}[/]')
        self.refresh()

    # step_done: plain mode prints one line per finished step
    def step_done(self, step: Step) -> None:
        if not self.live_on and not self.quiet:
            self.console.print(f'  {step.status:<7} {step.label:<17} {step.text} {step.detail}', markup=False)
        self.refresh()

    # end_group: commit the finished block to scrollback
    def end_group(self, freed: int) -> None:
        if self.live_on:
            self.console.print(self._panel())
        self.steps = []
        self.title = ''
        self.progress.update(self.task, advance=1, freed=_fmt(freed))
        self.refresh()

# Ctx: everything one run shares
@dataclass
class Ctx:
    conn: sqlite3.Connection
    plan: dict
    plan_path: str
    dry: bool
    display: Display
    stats: Counter
    failures: list[tuple[str, str]]

# _save_plan: write the plan atomically so an interrupted run never leaves a half-written file
def _save_plan(ctx: Ctx) -> None:
    ctx.plan['updated_at'] = time.time()
    tmp = f'{ctx.plan_path}.tmp'
    with open(tmp, 'w') as f:
        json.dump(ctx.plan, f, indent=2)
    os.replace(tmp, ctx.plan_path)

# _run: execute one step (or just mark it in a dry run), stamp the plan entry and report
def _run(ctx: Ctx, step: Step, fn, entry: dict | None = None, slot: str = '') -> str:
    step.status = 'running'
    ctx.display.refresh()
    if ctx.dry:
        step.status = 'would'
    else:
        step.status, step.detail = fn()
        if entry is not None:
            entry.setdefault('applied', {})[slot] = f'{step.status}: {step.detail}' if step.detail else step.status
            _save_plan(ctx)
    if step.status == 'failed':
        ctx.failures.append((f'{step.label} {step.text}', step.detail))
    ctx.display.step_done(step)
    return step.status

# _skip: mark a step as not run, with the reason
def _skip(ctx: Ctx, step: Step, reason: str) -> None:
    step.status, step.detail = 'skipped', reason
    ctx.display.step_done(step)

# _done: whether a plan entry slot was already applied in an earlier run
def _done(entry: dict, slot: str) -> bool:
    return entry.get('applied', {}).get(slot, '').startswith('ok')

# _db_path: stored path for a video id, None when there is no row
def _db_path(conn: sqlite3.Connection, vid: int) -> str | None:
    row = conn.execute('SELECT path FROM videos WHERE id = ?', (vid,)).fetchone()
    return row[0] if row else None

# _remove: delete a file for good; a file that is already gone counts as done
def _remove(path: str) -> tuple[str, str]:
    if not os.path.lexists(path):
        return 'ok', 'already gone'
    try:
        os.remove(path)
    except OSError as e:
        return 'failed', str(e)
    return 'ok', 'removed'

# _purge: delete a video's rows, children first, in one transaction (same order as matcha's delete_video_record)
def _purge(conn: sqlite3.Connection, vid: int) -> tuple[str, str]:
    try:
        with conn:
            rows = sum(conn.execute(f'DELETE FROM {t} WHERE video_id = ?', (vid,)).rowcount for t in _VIDEO_TABLES)
            rows += sum(
                conn.execute(f'DELETE FROM {t} WHERE video_a_id = ? OR video_b_id = ?', (vid, vid)).rowcount
                for t in _PAIR_TABLES
            )
            rows += conn.execute('DELETE FROM videos WHERE id = ?', (vid,)).rowcount
    except sqlite3.Error as e:
        return 'failed', str(e)
    return 'ok', f'{rows} rows across {len(_VIDEO_TABLES) + len(_PAIR_TABLES) + 1} tables'

# _rename_file: rename on disk
def _rename_file(src: str, dst: str) -> tuple[str, str]:
    try:
        os.rename(src, dst)
    except OSError as e:
        return 'failed', str(e)
    return 'ok', 'renamed'

# _update_path: point the video row at its new path
def _update_path(conn: sqlite3.Connection, vid: int, dst: str) -> tuple[str, str]:
    try:
        with conn:
            n = conn.execute('UPDATE videos SET path = ? WHERE id = ?', (dst, vid)).rowcount
    except sqlite3.Error as e:
        return 'failed', str(e)
    return ('ok', 'row updated') if n == 1 else ('failed', 'no such video row')

# _dismiss: mark matches between the surviving videos as dismissed, adding the column if an old db lacks it
def _dismiss(conn: sqlite3.Connection, ids: list[int]) -> tuple[str, str]:
    marks = ','.join('?' * len(ids))
    try:
        with conn:
            if 'dismissed' not in {r[1] for r in conn.execute('PRAGMA table_info(matches)')}:
                conn.execute('ALTER TABLE matches ADD COLUMN dismissed INTEGER NOT NULL DEFAULT 0')
            n = conn.execute(
                f'UPDATE matches SET dismissed = 1 WHERE video_a_id IN ({marks}) AND video_b_id IN ({marks})', ids + ids
            ).rowcount
    except sqlite3.Error as e:
        return 'failed', str(e)
    return 'ok', f'{n} matches'

# _delete_entry: remove the file, then purge its rows (the db step only runs once the file step succeeded)
def _delete_entry(ctx: Ctx, f: dict, fstep: Step, dstep: Step, guard: str) -> None:
    if guard:
        _skip(ctx, fstep, guard), _skip(ctx, dstep, guard)
        return
    if _done(f, 'file'):
        _skip(ctx, fstep, 'done earlier')
    else:
        row = _db_path(ctx.conn, f['id'])
        if row != f['path']:
            reason = 'not in db' if row is None else 'path differs from db, plan is stale'
            _skip(ctx, fstep, reason), _skip(ctx, dstep, reason)
            return
        if _run(ctx, fstep, lambda: _remove(f['path']), f, 'file') == 'failed':
            _skip(ctx, dstep, 'kept: file step failed')
            return
        if fstep.detail != 'already gone':
            ctx.stats['freed'] += f.get('size', 0)
    if _done(f, 'db'):
        _skip(ctx, dstep, 'done earlier')
    elif _run(ctx, dstep, lambda: _purge(ctx.conn, f['id']), f, 'db') in ('ok', 'would'):
        ctx.stats['deleted'] += 1

# _rename_entry: rename the file, then update its row; undo the rename if the row update fails
def _rename_entry(ctx: Ctx, f: dict, fstep: Step, dstep: Step, guard: str) -> None:
    src, dst = f['path'], f['new_path']
    if guard:
        _skip(ctx, fstep, guard), _skip(ctx, dstep, guard)
        return
    if _done(f, 'file'):
        _skip(ctx, fstep, 'done earlier')
    else:
        row = _db_path(ctx.conn, f['id'])
        taken = os.path.lexists(dst) or ctx.conn.execute('SELECT 1 FROM videos WHERE path = ?', (dst,)).fetchone()
        if row != src:
            reason = 'not in db' if row is None else 'path differs from db, plan is stale'
            _skip(ctx, fstep, reason), _skip(ctx, dstep, reason)
            return
        if not os.path.lexists(src):
            _skip(ctx, fstep, 'source file missing'), _skip(ctx, dstep, 'source file missing')
            return
        if taken:
            fstep.status, fstep.detail = 'failed', 'target name already exists'
            ctx.failures.append((f'{fstep.label} {fstep.text}', fstep.detail))
            ctx.display.step_done(fstep)
            _skip(ctx, dstep, 'kept: file step failed')
            return
        if _run(ctx, fstep, lambda: _rename_file(src, dst), f, 'file') == 'failed':
            _skip(ctx, dstep, 'kept: file step failed')
            return
    if _done(f, 'db'):
        _skip(ctx, dstep, 'done earlier')
        return
    status = _run(ctx, dstep, lambda: _update_path(ctx.conn, f['id'], dst), f, 'db')
    if status in ('ok', 'would'):
        ctx.stats['renamed'] += 1
    elif status == 'failed':
        undo = Step('file', 'undo rename', f'{os.path.basename(dst)} -> {os.path.basename(src)}')
        ctx.display.steps.append(undo)
        _run(ctx, undo, lambda: _rename_file(dst, src), f, 'file')
        f['applied']['file'] = 'failed: db update failed, rename undone'
        _save_plan(ctx)

# _group_done: every delete and rename already applied and the survivors already dismissed
def _group_done(group: dict) -> bool:
    files = group['files']
    survivors = [f for f in files if f['action'] != 'delete']
    ops = [f for f in files if f['action'] in ('delete', 'rename')]
    return all(_done(f, 'file') and _done(f, 'db') for f in ops) and (len(survivors) < 2 or group.get('dismissed', False))

# _apply_group: build the group's steps, then run deletes, renames and the dismissal in that order
def _apply_group(ctx: Ctx, group: dict, title: str) -> None:
    files = group['files']
    survivors = [f for f in files if f['action'] != 'delete']
    # never delete the last copy: some surviving file must be on disk, before or after its rename
    alive = [f for f in survivors if os.path.lexists(f['path']) or os.path.lexists(f.get('new_path', ''))]
    guard = '' if alive else 'no surviving file on disk, refusing to delete'

    name = lambda f: os.path.basename(f['path'])
    steps: list[Step] = [Step('info', 'keep', name(f), 'info') for f in files if f['action'] == 'keep']
    pairs: list[tuple[dict, Step, Step]] = []
    for f in files:
        if f['action'] == 'delete':
            pairs.append((f, Step('file', 'delete', f'{name(f)} ({_fmt(f.get("size", 0))})'), Step('db', 'purge', name(f))))
        elif f['action'] == 'rename':
            pairs.append((f, Step('file', 'rename', f'{name(f)} -> {os.path.basename(f["new_path"])}'), Step('db', 'update', 'videos.path')))
    pairs.sort(key=lambda p: p[0]['action'] != 'delete')  # deletes first so a freed name can be reused
    for _, fstep, dstep in pairs:
        steps += [fstep, dstep]
    dismiss = Step('db', 'dismiss', f'matches between {len(survivors)} survivors')
    if len(survivors) >= 2:
        steps.append(dismiss)

    ctx.display.start_group(title, steps)
    ctx.stats['kept'] += sum(f['action'] == 'keep' for f in files)
    for f, fstep, dstep in pairs:
        (_delete_entry if f['action'] == 'delete' else _rename_entry)(ctx, f, fstep, dstep, guard)
    if len(survivors) >= 2:
        if group.get('dismissed'):
            _skip(ctx, dismiss, 'done earlier')
        elif guard:
            _skip(ctx, dismiss, guard)
        elif _run(ctx, dismiss, lambda: _dismiss(ctx.conn, [f['id'] for f in survivors])) == 'ok':
            group['dismissed'] = True
            _save_plan(ctx)
    ctx.display.end_group(ctx.stats['freed'])


# _pending: (deletes, renames, bytes) still to do across the plan
def _pending(plan: dict) -> tuple[int, int, int]:
    todo = [f for g in plan['groups'] for f in g['files'] if f['action'] in ('delete', 'rename') and not (_done(f, 'file') and _done(f, 'db'))]
    deletes = [f for f in todo if f['action'] == 'delete']
    return len(deletes), len(todo) - len(deletes), sum(f.get('size', 0) for f in deletes)

# run: validate, optionally confirm, then apply every group; returns the process exit code
def run(directory: str, plan_path: str | None = None, *, apply: bool = False, yes: bool = False,
        quiet: bool = False, console: Console | None = None) -> int:
    console = console or Console(highlight=False)
    directory = os.path.abspath(directory)
    if not os.path.isdir(directory):
        raise SystemExit(f'library {directory!r} is not a directory (drive not mounted?)')
    db_path = os.path.join(directory, '.matcha', 'index.db')
    plan_path = plan_path or os.path.join(directory, '.matcha', 'review_plan.json')
    for path, what in ((db_path, 'index'), (plan_path, 'plan')):
        if not os.path.exists(path):
            raise SystemExit(f'no {what} at {path!r}')
    with open(plan_path) as f:
        plan = json.load(f)
    if plan.get('version') != 1:
        raise SystemExit(f'unsupported plan version {plan.get("version")!r}')

    deletes, renames, size = _pending(plan)
    console.print(f'plan: [bold]{deletes}[/] deletes ({_fmt(size)}), [bold]{renames}[/] renames pending' + ('' if apply else '  [cyan](dry run, pass --apply to execute)[/]'))
    if not deletes and not renames:
        console.print('nothing to apply')
        return 0
    if apply and not yes and input(f'Permanently delete {deletes} files? Type yes to continue: ').strip() != 'yes':
        console.print('aborted')
        return 1

    conn = sqlite3.connect(db_path)
    conn.execute('PRAGMA foreign_keys=ON')
    conn.execute('PRAGMA busy_timeout=5000')
    groups = plan['groups']
    display = Display(console, quiet=quiet, total=len(groups))
    ctx = Ctx(conn, plan, plan_path, not apply, display, Counter(), [])
    with display:
        for n, group in enumerate(groups, 1):
            title = f'Group {n}/{len(groups)}  ({len(group["files"])} files)'
            if _group_done(group):
                ctx.stats['groups_skipped'] += 1
                display.start_group(title + '  already applied', [])
                display.end_group(ctx.stats['freed'])
            else:
                _apply_group(ctx, group, title)

    verb = 'would be' if ctx.dry else 'were'
    console.print(f'\n[bold green]{_fmt(ctx.stats["freed"])}[/] freed: {ctx.stats["deleted"]} deleted, '
                  f'{ctx.stats["renamed"]} renamed, {ctx.stats["kept"]} kept '
                  f'({"dry run, nothing changed" if ctx.dry else "changes " + verb + " applied"})')
    for what, reason in ctx.failures:
        console.print(f'[red]failed[/] {what}: {reason}', markup=True)
    return 1 if ctx.failures else 0
