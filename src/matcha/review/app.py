'''
MATCHA: Streamlit app for reviewing duplicate groups.
'''

# Import external dependencies
import os
import streamlit as st

# Import internal matcha objects
from matcha.db import init_schema
from matcha.groups import SORTS, Group, find_groups, matches_version, paginate, sort_groups, video_names

# PER_PAGE: groups shown per page, so a large library only ever renders a screenful
PER_PAGE = 25
# NAMES_SHOWN: file names listed on each group card
NAMES_SHOWN = 3


# _db_path: index database for the library the app was launched on
def _db_path() -> str:
    return os.path.join(os.environ['MATCHA_REVIEW_DIR'], '.matcha', 'index.db')


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
                st.session_state.open_group = group.key
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


# _detail_view: placeholder until the group detail view lands
def _detail_view(group_key: int) -> None:
    st.title(f'Group {group_key}')
    st.info('Group detail is not built yet.')
    if st.button('Back'):
        st.session_state.open_group = None
        st.rerun()


# _main: route between the list and detail views
def _main() -> None:
    st.set_page_config(page_title='Matcha review', layout='wide')
    db_path = _db_path()
    init_schema(db_path)
    if st.session_state.get('open_group') is None:
        _list_view(db_path)
    else:
        _detail_view(st.session_state.open_group)


_main()
