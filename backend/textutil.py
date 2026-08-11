"""Shared text helpers."""
from __future__ import annotations

import re

# Articles MovieLens moves to the end ("Matrix, The" → "The Matrix").
_ARTICLES = {"The", "A", "An", "La", "Le", "Les", "L'", "Der", "Die", "Das",
             "El", "Il", "Lo", "Gli", "I", "Un", "Une", "Os", "As"}


def clean_title(title: str) -> str:
    """"Matrix, The (1999)" → "The Matrix"; keeps a trailing alt-title paren."""
    t = re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()
    m = re.match(r"^(.*),\s+([\w']+)(\s*\([^)]*\))?$", t)
    if m and m.group(2) in _ARTICLES:
        art = m.group(2)
        sep = "" if art.endswith("'") else " "
        t = f"{art}{sep}{m.group(1)}{m.group(3) or ''}"
    return t
