"""
This part of code is borrowed from https://github.com/RimoChan/i7h
with adaptations (replace \w+ with [a-zA-Z]+, where digits and underscores
are not included).

License: None
"""

import re


def _f(r):
    t = r.group()
    if len(t) <= 2:
        return t
    return f"{t[0]}{len(t)-2}{t[-1]}"


def i18n(s: str) -> str:
    s = re.sub(r"[a-zA-Z]+", _f, s)
    return s


if __name__ == "__main__":
    text = """Alice 99___9 was beginning to get very tired of sitting by her sister
on the bank, and of having nothing to do:  once or twice she had
peeped into the book her sister was reading, but it had no
pictures or conversations in it, `and what is the use of a book,'
thought Alice `without pictures or conversation?'"""
    print(i18n(text))
