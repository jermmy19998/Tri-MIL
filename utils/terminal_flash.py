"""
terminal_flash.py

Minimal terminal flashing utility.

Only ONE public function:
    flash(text, fg, bg, times, interval, bold)

Features:
- ANSI colored flashing text
- Configurable foreground / background
- Configurable flashing times and interval
- Bold text support
- No external dependencies

Compatible with:
- Linux / macOS
- SSH / VSCode / PyCharm / Docker / Slurm
"""

import sys
import time


def flash(
    text: str,
    fg: str = "black",
    bg: str = "red",
    times: int = 3,
    interval: float = 0.25,
    bold: bool = True,
):
    """
    Flash a message in terminal.

    Parameters
    ----------
    text : str
        Message to display
    fg : str
        Foreground color:
        black / white / red / green / yellow / blue
    bg : str
        Background color:
        red / green / yellow / blue
    times : int
        Number of flash cycles
    interval : float
        Interval between flashes (seconds)
    bold : bool
        Whether to use bold text
    """

    # ANSI foreground color codes
    fg_codes = {
        "black": "30",
        "red": "31",
        "green": "32",
        "yellow": "33",
        "blue": "34",
        "white": "37",
    }

    # ANSI background color codes
    bg_codes = {
        "red": "41",
        "green": "42",
        "yellow": "43",
        "blue": "44",
    }

    style = []

    # Enable bold font if requested
    if bold:
        style.append("1")

    style.append(fg_codes.get(fg, "37"))
    style.append(bg_codes.get(bg, "40"))

    # ANSI escape prefix and reset
    prefix = "\033[" + ";".join(style) + "m"
    reset = "\033[0m"

    msg = f" {text} "
    blank = " " * len(msg)

    # Flash loop
    for _ in range(times):
        sys.stdout.write("\r" + prefix + msg + reset)
        sys.stdout.flush()
        time.sleep(interval)

        sys.stdout.write("\r" + blank)
        sys.stdout.flush()
        time.sleep(interval)

    # Final stable display
    print(prefix + msg + reset)
