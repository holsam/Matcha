'''
Shared interactive-input helpers used by multiple matcha commands
'''

import io, os, sys, termios, threading, tty


def watch_for_quit(stop_event: threading.Event):
    '''
    Background thread that sets stop_event when 'q' is pressed.

    Puts stdin into raw (unbuffered, no-echo) mode so keypresses are
    received immediately without the user pressing Enter. Restores the
    original terminal settings on exit regardless of how it ends.
    '''
    fd = sys.stdin.fileno()
    try:
        old_settings = termios.tcgetattr(fd)
    except (termios.error, io.UnsupportedOperation):
        # stdin is not a tty (e.g. in tests or piped input) — skip listener
        return
    try:
        tty.setraw(fd)
        while not stop_event.is_set():
            # os.read is non-blocking after setraw; use select to avoid busy-wait
            import select
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                ch = os.read(fd, 1)
                if ch in (b'q', b'Q'):
                    stop_event.set()
                    break
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)