'''
MATCHA: Streamlit app for reviewing duplicate groups.
'''

# Import external dependencies
import os
import streamlit as st
from datetime import datetime

# Import internal matcha objects
from matcha import actions
from matcha.db import init_schema
from matcha.groups import (
    SORTS, Group, VideoRow, find_groups, load_pairs, load_videos, matches_version,
    paginate, resolve_group, sort_groups, video_names,
)
from matcha.review import media

# PER_PAGE: groups shown per page, so a large library only ever renders a screenful
PER_PAGE = 25
# NAMES_SHOWN: file names listed on each group card
NAMES_SHOWN = 3


# _load_groups: all groups, cached until the matches table changes
@st.cache_data(show_spinner='Grouping matches...')
def _load_groups(db_path: str, version: tuple[int, int, int]) -> list[Group]:
    return find_groups(db_path)


# _card: one group summary, database-only (no stat, no ffmpeg)
def _card(db_path: str, group: Group) -> None:
    with st.container(border=True):
        left, right = st.columns([5, 1], vertical_alignment='center')
        with left:
            kind = 'duplicate + subclip' if group.has_subclip else 'duplicate'
            st.markdown(f'**{len(group.video_ids)} videos** · {kind} · best match {group.max_confidence:.0%} · {group.total_duration / 60:.0f} min total')
            names = video_names(db_path, list(group.video_ids[:NAMES_SHOWN]))
            extra = len(group.video_ids) - NAMES_SHOWN
            listed = ', '.join(names[v] for v in group.video_ids[:NAMES_SHOWN] if v in names)
            st.caption(listed + (f' and {extra} more' if extra > 0 else ''))
        with right:
            if st.button('Open', key=f'open-{group.key}', width='stretch'):
                st.session_state.open_group = group.video_ids
                st.rerun()


# _list_view: sortable, paginated group list
def _list_view(db_path: str) -> None:
    groups = _load_groups(db_path, matches_version(db_path))
    st.title('Matcha review')
    if not groups:
        st.info('No duplicate groups to review.')
        return

    top, mid = st.columns([2, 3], vertical_alignment='bottom')
    sort = top.selectbox('Sort by', list(SORTS), format_func=lambda k: SORTS[k][0], key='sort')
    shown, page, pages = paginate(sort_groups(groups, sort), st.session_state.get('page', 0), PER_PAGE)
    mid.caption(f'{len(groups):,} groups · page {page + 1} of {pages}')

    for group in shown:
        _card(db_path, group)

    prev, _, nxt = st.columns([1, 4, 1])
    if prev.button('Previous', disabled=page == 0, width='stretch'):
        st.session_state.page = page - 1
        st.rerun()
    if nxt.button('Next', disabled=page >= pages - 1, width='stretch'):
        st.session_state.page = page + 1
        st.rerun()


# _library: root directory of the library the app was launched on
def _library() -> str:
    return os.environ['MATCHA_REVIEW_DIR']


# _flash: queue a message to show after the next rerun
def _flash(kind: str, message: str) -> None:
    st.session_state.setdefault('flash', []).append((kind, message))


# _show_flash: pop queued messages as toasts
def _show_flash() -> None:
    for kind, message in st.session_state.pop('flash', []):
        st.toast(message, icon=':material/error:' if kind == 'error' else ':material/check:')


# _close_group: return to the list view
def _close_group() -> None:
    st.session_state.open_group = None
    st.session_state.pending = None


# _report: turn a DeleteResult into flash messages
def _report(result: actions.DeleteResult) -> None:
    if result.deleted:
        _flash('ok', f'Moved {len(result.deleted)} video(s) to the trash.')
    for vid, why in result.failed.items():
        _flash('error', f'Video {vid}: {why}')


# _rename_popover: inline rename control for one video
def _rename_popover(video: VideoRow) -> None:
    with st.popover('Rename', width='stretch'):
        name = st.text_input('New name', value=os.path.basename(video.path), key=f'name-{video.id}')
        if st.button('Apply', key=f'rename-{video.id}'):
            try:
                actions.rename_video(_library(), video.id, name)
                _flash('ok', f'Renamed to {name}.')
            except (ValueError, OSError) as exc:
                _flash('error', str(exc))
            st.rerun()


# _video_row: one video with its on-disk facts, loaded only because its group is open
def _video_row(video: VideoRow, group_ids: list[int]) -> None:
    info = media.file_info(video.path)
    thumb, details, controls = st.columns([2, 5, 2])
    with thumb:
        shot = media.thumbnail(os.path.join(_library(), '.matcha'), video.id, video.path, video.duration) if info else None
        if shot:
            st.image(shot)
        else:
            st.caption('no preview')
    with details:
        st.checkbox(f'**{os.path.basename(video.path)}**', key=f'sel-{video.id}', disabled=info is None)
        st.caption(video.path)
        if info is None:
            st.warning('File is missing from disk. Delete it to drop it from the index.')
        else:
            when = datetime.fromtimestamp(info.mtime).strftime('%Y-%m-%d %H:%M')
            st.write(f'{media.human_size(info.size)} · {media.human_duration(video.duration)} · modified {when}')
            if st.toggle('Technical details', key=f'tech-{video.id}'):
                p = media.probe(video.path)
                if p is None:
                    st.caption('ffprobe could not read this file.')
                else:
                    size = f'{p.width}x{p.height}' if p.width and p.height else 'unknown size'
                    rate = f'{p.bitrate / 1e6:.1f} Mbit/s' if p.bitrate else 'unknown bitrate'
                    st.caption(f'{size} · {p.codec or "unknown codec"} · {rate}')
            if st.toggle('Preview', key=f'prev-{video.id}'):
                st.video(video.path)
    with controls:
        if st.button('Keep only this', key=f'keep-{video.id}', width='stretch', disabled=len(group_ids) < 2):
            others = [v for v in group_ids if v != video.id]
            st.session_state.pending = {'ids': others, 'label': f'Keep {os.path.basename(video.path)} and trash the other {len(others)}.'}
            st.rerun()
        _rename_popover(video)


# _confirm_delete: explicit confirmation before anything reaches the trash
def _confirm_delete(videos: dict[int, VideoRow]) -> None:
    pending = st.session_state.pending
    with st.container(border=True):
        st.warning(pending['label'])
        st.markdown('\n'.join(f'- {videos[v].path}' for v in pending['ids'] if v in videos))
        yes, no, _ = st.columns([1, 1, 4])
        if yes.button('Move to trash', type='primary', width='stretch'):
            _report(actions.delete_videos(_library(), pending['ids']))
            st.session_state.pending = None
            st.rerun()
        if no.button('Cancel', width='stretch'):
            st.session_state.pending = None
            st.rerun()


# _pairs_table: pairwise similarity and match type within the group
def _pairs_table(db_path: str, videos: dict[int, VideoRow]) -> None:
    pairs = load_pairs(db_path, list(videos))
    st.subheader('Pairwise matches')
    st.caption('A subclip match means the shorter video is contained in the longer one.')
    st.dataframe(
        [
            {
                'Video A': os.path.basename(videos[p.a].path),
                'Video B': os.path.basename(videos[p.b].path),
                'Type': p.match_type,
                'Similarity': f'{p.confidence:.0%}',
            }
            for p in pairs
        ],
        hide_index=True,
        width='stretch',
    )


# _detail_view: every video in one group with the actions available on them
def _detail_view(db_path: str, group: Group) -> None:
    ids = list(group.video_ids)
    videos = {v.id: v for v in load_videos(db_path, ids)}

    back, dismiss, _ = st.columns([1, 2, 5])
    if back.button('Back', width='stretch'):
        _close_group()
        st.rerun()
    if dismiss.button('Not duplicates, dismiss', width='stretch'):
        actions.dismiss_group(_library(), ids)
        _flash('ok', 'Group dismissed.')
        _close_group()
        st.rerun()

    st.title(f'{len(ids)} matched videos')
    if st.session_state.get('pending'):
        _confirm_delete(videos)

    for video in videos.values():
        with st.container(border=True):
            _video_row(video, ids)

    selected = [v for v in videos if st.session_state.get(f'sel-{v}')]
    if st.button(f'Delete selected ({len(selected)})', disabled=not selected, type='primary'):
        st.session_state.pending = {'ids': selected, 'label': f'Move {len(selected)} video(s) to the trash.'}
        st.rerun()

    _pairs_table(db_path, videos)


# _main: route between the list and detail views
def _main() -> None:
    st.set_page_config(page_title='Matcha review', layout='wide')
    db_path = os.path.join(_library(), '.matcha', 'index.db')
    init_schema(db_path)
    _show_flash()

    open_ids = st.session_state.get('open_group')
    if open_ids is None:
        _list_view(db_path)
        return
    # membership changes after deletes, so find the group again instead of trusting a stored key
    group = resolve_group(_load_groups(db_path, matches_version(db_path)), open_ids)
    if group is None:
        _close_group()
        _list_view(db_path)
        return
    st.session_state.open_group = group.video_ids
    _detail_view(db_path, group)


_main()
