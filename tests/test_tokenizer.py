import pytest

from mini_llm.tokenizer import CharTokenizer


def test_vocab_is_sorted_unique_characters() -> None:
    tok = CharTokenizer.from_text("hello")
    assert tok.vocab == ["e", "h", "l", "o"]
    assert tok.vocab_size == 4


def test_encode_maps_each_character_to_its_index() -> None:
    tok = CharTokenizer.from_text("hello")
    assert tok.encode("hello") == [1, 0, 2, 2, 3]


def test_decode_is_the_inverse_of_encode() -> None:
    text = "MONSIEUR JOURDAIN. — Quoi ? quand je dis : « Nicole » ...\n"
    tok = CharTokenizer.from_text(text)
    assert tok.decode(tok.encode(text)) == text


def test_encode_rejects_unknown_characters() -> None:
    tok = CharTokenizer.from_text("abc")
    with pytest.raises(ValueError):
        tok.encode("abz")


def test_save_and_load_roundtrip(tmp_path) -> None:
    tok = CharTokenizer.from_text("Le Bourgeois gentilhomme")
    path = tmp_path / "tokenizer.json"
    tok.save(path)
    loaded = CharTokenizer.load(path)
    assert loaded.vocab == tok.vocab
    assert loaded.encode("Bourgeois") == tok.encode("Bourgeois")
