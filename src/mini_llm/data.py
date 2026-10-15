"""Text cleaning for Project Gutenberg books.

Raw Gutenberg files contain a licence header and footer, footnote markers and
irregular whitespace. A language model learns everything it sees, so we remove
what is not part of the plays themselves.
"""

from __future__ import annotations

import re

START_MARKER = "*** START OF"
END_MARKER = "*** END OF"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Keep only the book content, between the START and END markers.

    The START marker line and the END marker line are both removed, as well as
    everything before the first one and after the second one. The result is
    stripped of leading and trailing whitespace.

    Raises ValueError if one of the markers is missing.
    """
    start = text.find(START_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1:
        raise ValueError("Gutenberg START or END marker not found")
    return text[text.find("\n", start) + 1 : end].strip()


def remove_footnote_markers(text: str) -> str:
    """Remove numeric footnote references such as "[1]" or "[23]".

    Brackets that contain something else, e.g. a stage direction like
    "[à part]", must be kept.
    """
    return re.sub(r"\[\d+\]", "", text)


def normalize_whitespace(text: str) -> str:
    """Make whitespace regular.

    - Windows line endings "\\r\\n" become "\\n".
    - Spaces at the end of each line are removed.
    - Three or more consecutive newlines become exactly two (one blank line).
    - Leading and trailing whitespace of the whole text is removed.
    """
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def clean_gutenberg_text(text: str) -> str:
    """Full cleaning pipeline: apply the three functions above, in a sensible order."""
    text = strip_gutenberg_boilerplate(text)
    text = remove_footnote_markers(text)
    return normalize_whitespace(text)
