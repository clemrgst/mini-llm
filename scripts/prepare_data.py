"""Download Molière's plays from Project Gutenberg, clean them and build the tokenizer.

Usage (from the project root):
    uv run python scripts/prepare_data.py

Outputs (not tracked by Git):
    data/raw/<id>.txt     raw downloads, cached so we only download once
    data/moliere.txt      the cleaned training corpus
    data/tokenizer.json   the character vocabulary
"""

from __future__ import annotations

import urllib.request
from pathlib import Path

from mini_llm.data import clean_gutenberg_text
from mini_llm.tokenizer import CharTokenizer

# French editions of Molière's plays on Project Gutenberg: {ebook id: title}
PLAYS = {
    3645: "L'Étourdi ou les contre-temps",
    5318: "Les Précieuses ridicules",
    5644: "Sganarelle, ou le Cocu imaginaire",
    5130: "Don Juan, ou le Festin de pierre",
    5178: "Le Mariage forcé",
    20498: "Le Médecin malgré lui",
    6318: "L'Avare",
}
URL = "https://www.gutenberg.org/cache/epub/{id}/pg{id}.txt"

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"


def download(book_id: int) -> str:
    """Return the raw text of a book, downloading it only if it is not cached yet."""
    path = RAW_DIR / f"{book_id}.txt"
    if not path.exists():
        print(f"  downloading {book_id}...")
        request = urllib.request.Request(
            URL.format(id=book_id), headers={"User-Agent": "mini-llm (educational project)"}
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            path.write_bytes(response.read())
    return path.read_text(encoding="utf-8-sig")


def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    plays = []
    for book_id, title in PLAYS.items():
        print(f"{title} ({book_id})")
        plays.append(clean_gutenberg_text(download(book_id)))

    corpus = "\n\n".join(plays) + "\n"
    (DATA_DIR / "moliere.txt").write_text(corpus, encoding="utf-8")

    tokenizer = CharTokenizer.from_text(corpus)
    tokenizer.save(DATA_DIR / "tokenizer.json")

    print()
    print(f"Corpus:     {len(corpus):,} characters, {len(PLAYS)} plays")
    print(f"Vocabulary: {tokenizer.vocab_size} distinct characters")
    print("".join(tokenizer.vocab))


if __name__ == "__main__":
    main()
