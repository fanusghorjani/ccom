import json
import re
from pathlib import Path

# === Config ===
IN_DIR = Path("corpus") / "clean_txt"
OUT_DIR = Path("processed") / "chunks"
OUT_JSONL = OUT_DIR / "chunks.jsonl"

TARGET_WORDS = 400       # chunk size target
MIN_WORDS = 200          # merge small tail chunks
MAX_WORDS = 520          # hard cap (prevents huge chunks)
OVERLAP_WORDS = 60       # overlap between chunks (0 to disable)

# Paragraph splitting: treat 2+ newlines as paragraph boundary
PARA_SPLIT_RE = re.compile(r"\n{2,}")

# TOC removal heuristics
TOC_START_RE = re.compile(r"^\s*Contents\s*$", re.IGNORECASE)
DOT_LEADER_RE = re.compile(r"\.{5,}")
ONLY_DIGITS_RE = re.compile(r"^\s*\d+\s*$")

# Line-level cleanup heuristics (TOC-ish lines)
TOC_TRAILING_PAGENUM_RE = re.compile(r"^(.*)\s{1,}\d{1,4}\s*$")  # title ... 123


def strip_toc_block(text: str) -> str:
    """
    Remove a typical table-of-contents block:
    starts at a line 'Contents' and continues while lines look TOC-ish
    (dot leaders / page-number lines / empty lines).
    """
    lines = text.split("\n")
    out = []
    i = 0
    while i < len(lines):
        if TOC_START_RE.match(lines[i]):
            i += 1
            skipped = 0
            while i < len(lines) and skipped < 500:
                ln = lines[i].strip()
                if not ln:
                    i += 1
                    skipped += 1
                    continue
                if DOT_LEADER_RE.search(ln) or ONLY_DIGITS_RE.match(ln):
                    i += 1
                    skipped += 1
                    continue
                # still TOC-ish if it ends with a small page number and has dots
                if re.search(r"\s\d{1,4}$", ln) and ("." in ln):
                    i += 1
                    skipped += 1
                    continue
                # stop skipping when normal text starts
                break
            continue

        out.append(lines[i])
        i += 1

    return "\n".join(out)


def normalize_ws(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # Remove common TOC artifacts line-by-line
    cleaned_lines = []
    for line in s.split("\n"):
	        # Drop dot-leader TOC lines anywhere (e.g., "Chapter .... 123")
        if DOT_LEADER_RE.search(line):
            continue
        # drop pure page number lines
        if ONLY_DIGITS_RE.match(line):
            continue

        # drop lines dominated by dot leaders
        if DOT_LEADER_RE.search(line):
            stripped = re.sub(r"[.\s\d]", "", line)
            if len(stripped) == 0:
                continue

        # drop lines that are just "Endnotes .... 123" or similar TOC patterns
        m = TOC_TRAILING_PAGENUM_RE.match(line.strip())
        if m and ("." in line or "Contents" in line or "Endnotes" in line):
            left = m.group(1).strip()
            if left and len(left) > 3:
                cleaned_lines.append(left)
            continue

        cleaned_lines.append(line)

    s = "\n".join(cleaned_lines)

    # normalize internal whitespace but keep paragraph breaks
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def split_paragraphs(text: str) -> list[str]:
    text = normalize_ws(text)
    return [p.strip() for p in PARA_SPLIT_RE.split(text) if p.strip()]


def words(s: str) -> list[str]:
    return s.split()


def make_chunks_from_paras(paras: list[str]) -> list[str]:
    chunks: list[str] = []
    cur_words: list[str] = []
    cur_text_parts: list[str] = []

    def flush_chunk():
        nonlocal cur_words, cur_text_parts
        if not cur_text_parts:
            return
        chunk_text = "\n\n".join(cur_text_parts).strip()
        chunks.append(chunk_text)
        cur_words = []
        cur_text_parts = []

    for p in paras:
        p_words = words(p)

        # If a single paragraph is huge, split by words (fallback)
        if len(p_words) > MAX_WORDS:
            flush_chunk()
            for i in range(0, len(p_words), TARGET_WORDS):
                piece = " ".join(p_words[i:i + TARGET_WORDS]).strip()
                if piece:
                    chunks.append(piece)
            continue

        # If adding paragraph would exceed MAX_WORDS, flush first
        if len(cur_words) + len(p_words) > MAX_WORDS:
            flush_chunk()

        cur_text_parts.append(p)
        cur_words.extend(p_words)

        # If we hit target, flush
        if len(cur_words) >= TARGET_WORDS:
            flush_chunk()

    flush_chunk()

    # Merge too-small tail into previous if possible
    if len(chunks) >= 2:
        last_wc = len(words(chunks[-1]))
        if last_wc < MIN_WORDS:
            merged = chunks[-2].strip() + "\n\n" + chunks[-1].strip()
            chunks = chunks[:-2] + [merged]

    return chunks


def add_overlap(chunks: list[str], overlap_words: int) -> list[str]:
    if overlap_words <= 0 or len(chunks) <= 1:
        return chunks

    new_chunks = [chunks[0]]
    prev_words = words(chunks[0])

    for i in range(1, len(chunks)):
        cur = chunks[i]
        cur_words = words(cur)

        overlap = prev_words[-overlap_words:] if len(prev_words) >= overlap_words else prev_words
        if overlap:
            cur = (" ".join(overlap) + "\n\n" + cur).strip()

        new_chunks.append(cur)
        prev_words = cur_words  # based on original previous chunk words
    return new_chunks


def main():
    OUT_DIR.mkdir(exist_ok=True)

    files = sorted(IN_DIR.glob("*.txt"))
    if not files:
        raise SystemExit(f"No .txt files found in {IN_DIR.resolve()}")

    total_chunks = 0
    with OUT_JSONL.open("w", encoding="utf-8") as out:
        for fp in files:
            doc_id = fp.stem
            text = fp.read_text(encoding="utf-8", errors="ignore")

            # 1) remove TOC block (if present)
            text = strip_toc_block(text)

            # 2) paragraph-aware chunking
            paras = split_paragraphs(text)
            chunks = make_chunks_from_paras(paras)
            chunks = add_overlap(chunks, OVERLAP_WORDS)

            for idx, ch in enumerate(chunks, start=1):
                wc = len(words(ch))
                rec = {
                    "doc_id": doc_id,
                    "chunk_id": f"{doc_id}_{idx:04d}",
                    "text": ch,
                    "word_count": wc,
                }
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                total_chunks += 1

    print("Chunking done.")
    print(f"Docs: {len(files)} | Chunks: {total_chunks}")
    print(f"Output: {OUT_JSONL.resolve()}")


if __name__ == "__main__":
    main()