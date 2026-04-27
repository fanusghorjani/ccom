import re
from pathlib import Path

IN_DIR = Path("corpus") / "raw_txt"
OUT_DIR = Path("corpus") / "clean_txt"
OUT_DIR.mkdir(exist_ok=True)

def fix_mojibake(s: str) -> str:
    try:
        if any(x in s for x in ["â€™", "â€œ", "â€", "Â", "Ã", "â€“", "â€”"]):
            s2 = s.encode("latin1", errors="ignore").decode("utf-8", errors="ignore")
            if s2:
                s = s2
    except Exception:
        pass

    repl = {
        "â€™": "’",
        "â€˜": "‘",
        "â€œ": "“",
        "â€": "”",
        "â€“": "–",
        "â€”": "—",
        "â€¦": "…",
        "Â": "",
        "Ã©": "é",
        "Ã¨": "è",
        "Ãª": "ê",
        "Ã¤": "ä",
        "Ã¶": "ö",
        "Ã¼": "ü",
        "ÃŸ": "ß",
    }
    for a, b in repl.items():
        s = s.replace(a, b)

    return s

def clean_text(s: str) -> str:
    s = fix_mojibake(s)

    s = s.replace("\r\n", "\n").replace("\r", "\n")

    # remove hyphenation line breaks
    s = re.sub(r"(\w)-\n(\w)", r"\1\2", s)

    # remove extra spaces
    s = re.sub(r"[ \t]+", " ", s)

    # remove standalone page numbers
    s = re.sub(r"\n\s*\d+\s*\n", "\n", s)

    # collapse excessive newlines
    s = re.sub(r"\n{3,}", "\n\n", s)

    return s.strip() + "\n"

def main():
    txt_files = sorted(IN_DIR.glob("*.txt"))
    if not txt_files:
        raise SystemExit(f"No .txt files found in {IN_DIR}")

    for fp in txt_files:
        raw = fp.read_text(encoding="utf-8", errors="ignore")
        cleaned = clean_text(raw)
        out_fp = OUT_DIR / fp.name
        out_fp.write_text(cleaned, encoding="utf-8")

    print(f"Cleaned {len(txt_files)} files.")

if __name__ == "__main__":
    main()