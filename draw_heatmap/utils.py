import shutil 


# -----------------------------
# Flash message
# -----------------------------
def flash_message(text, fg="black", bg="red", times=2, interval=0.25, bold=True):
    term_width = shutil.get_terminal_size().columns
    fg_codes = {
        "black": "30", "red": "31", "green": "32",
        "yellow": "33", "blue": "34", "white": "37",
    }
    bg_codes = {"red": "41", "green": "42", "yellow": "43", "blue": "44"}
    style = ["1"] if bold else []
    style.append(fg_codes.get(fg, "37"))
    style.append(bg_codes.get(bg, "40"))
    prefix = "\033[" + ";".join(style) + "m"
    reset = "\033[0m"

    pad = max((term_width - len(text)) // 2, 0)
    display_text = " " * pad + text + " " * (term_width - len(text) - pad)

    for _ in range(times):
        sys.stdout.write("\r" + prefix + display_text + reset)
        sys.stdout.flush()
        time.sleep(interval)
        sys.stdout.write("\r" + " " * term_width)
        sys.stdout.flush()
        time.sleep(interval)

    print("\r" + prefix + display_text + reset)
