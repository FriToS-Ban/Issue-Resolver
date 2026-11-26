"""retrieval/text.py — Code-aware tokenisation shared by every retrieval channel."""

from __future__ import annotations

import re

_IDENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SPLIT = re.compile(r"_+|(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# English filler + language keywords that carry no retrieval signal.
STOP = frozenset("""
a an the is it in on at of to and or but for with this that not be are was were has have had
from by as i we you they he she my our your their its me us them there here when how what why
which who whom can could should would will shall may might must do does did done get got
if else elif then than so such into out up down over under again more most some any all each
also just very only own same too now new old use used using like see seems seem one two
self cls def return import pass none null true false class lambda yield await async while
break continue global nonlocal assert del finally try except raise with var let const function
public private protected static void int str float bool dict list tuple set string number
issue bug error problem please thanks thank hi hello expected actual instead happens happen
works working work wrong fails failing fail broken expected behaviour behavior steps reproduce
version python file line code value values result results returns returned call called calls
""".split())

COMMON_CODE_NAMES = frozenset("""
get set run main init __init__ setup test tests helper helpers util utils data value values item
items key keys name names add update load save read write open close start stop process handle
to_dict from_dict __repr__ __str__ __eq__ __hash__ __post_init__ apply parse format check validate
""".split())


def split_identifier(word: str) -> list[str]:
    return [p.lower() for p in _SPLIT.split(word) if p]


def tokenize(text: str, keep_stop: bool = False) -> list[str]:
    """Full identifiers (lower-cased) plus their snake/camel sub-tokens."""
    out: list[str] = []
    for w in _IDENT.findall(text or ""):
        wl = w.lower()
        if len(wl) >= 2 and (keep_stop or wl not in STOP):
            out.append(wl)
        parts = split_identifier(w)
        if len(parts) > 1:
            out.extend(p for p in parts if len(p) >= 2 and (keep_stop or p not in STOP))
    return out


def stem(token: str) -> str:
    """Very light suffix stripping so 'expired'/'expiry'/'expires' meet."""
    for suf in ("ations", "ation", "ings", "ing", "ies", "ied", "ers", "er", "ed", "es", "s", "y"):
        if len(token) > len(suf) + 3 and token.endswith(suf):
            return token[: -len(suf)]
    return token
