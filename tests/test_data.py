import pytest

from mini_llm.data import (
    clean_gutenberg_text,
    normalize_whitespace,
    remove_footnote_markers,
    strip_gutenberg_boilerplate,
)

RAW_BOOK = (
    "The Project Gutenberg eBook of L'Avare\n"
    "Licence blah blah\n"
    "*** START OF THE PROJECT GUTENBERG EBOOK L'AVARE ***\n"
    "\n"
    "VALÈRE.\n"
    "Hé quoi ! charmante Élise[1], vous devenez mélancolique ?\n"
    "*** END OF THE PROJECT GUTENBERG EBOOK L'AVARE ***\n"
    "More licence text\n"
)


def test_strip_keeps_only_the_content_between_markers() -> None:
    assert strip_gutenberg_boilerplate(RAW_BOOK) == (
        "VALÈRE.\nHé quoi ! charmante Élise[1], vous devenez mélancolique ?"
    )


def test_strip_raises_when_a_marker_is_missing() -> None:
    with pytest.raises(ValueError):
        strip_gutenberg_boilerplate("A text without any Gutenberg marker")


def test_remove_footnote_markers_removes_numbers_in_brackets() -> None:
    assert remove_footnote_markers("Je vous vois[1] soupirer[23] !") == "Je vous vois soupirer !"


def test_remove_footnote_markers_keeps_other_brackets() -> None:
    assert remove_footnote_markers("HARPAGON, [à part].") == "HARPAGON, [à part]."


def test_normalize_whitespace() -> None:
    raw = "  Acte I  \r\nScène 1\n\n\n\nHARPAGON.   \nVoleur !\n\n"
    assert normalize_whitespace(raw) == "Acte I\nScène 1\n\nHARPAGON.\nVoleur !"


def test_clean_gutenberg_text_full_pipeline() -> None:
    assert clean_gutenberg_text(RAW_BOOK) == (
        "VALÈRE.\nHé quoi ! charmante Élise, vous devenez mélancolique ?"
    )
