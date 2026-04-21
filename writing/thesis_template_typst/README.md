# Thesis Template

Entry point is `main.typ`. All document metadata lives in `setup.typ`.

## Structure

`main.typ` chains show rules that wrap the remaining content in sequence:

```
the_body        → front matter (title page, executive summary, TOC) + chapters
the_bibliography → bibliography
the_appendix    → appendices (numbered A1, A2, …)
the_statement   → statement of authorship
```

Each function is defined in `include/template.typ` and exported via `*`.

## Adding content

- **Chapters**: add `#include "N-chaptername.typ"` inside `the_body` in `main.typ`
- **Appendices**: add `#include "X-appendix.typ"` inside `the_appendix` block
- **Metadata**: edit `setup.typ` — title, author, supervisor, dates, bib files, statements

## Configuration (`setup.typ`)

| Field | Purpose |
|---|---|
| `title` / `subtitle` | Title page |
| `author` | Name, email, affiliation, programme |
| `supervision.supervisor` | Supervisor name and affiliation |
| `supervision.practice_partner` | Omits the field if `none` |
| `date` | Submission date shown on title page |
| `meta.words` | Passed in automatically from `wordometer` |
| `meta.last_accessed` | Date shown before bibliography |
| `meta.jel` / `meta.keywords` | Omitted from abstract page if `none` |
| `bibliography_files` | Tuple of `.bib` files (relative to `include/`) |
| `statements.authorship` | Text of the statement of authorship |
| `statements.coi` / `statements.da` | Omitted from abstract page if `none` |
| `abstract` | Executive summary content |
| `acknowledgements` | Acknowledgements content |

## Word count

Per the Hertie guidelines, the word count includes all texts, footnotes, tables, and title pages; it excludes bibliography and appendices.

To exclude a passage from the word count, attach the `<no-wc>` label to it:

```typst
[This text is excluded. <no-wc>]
```

Deviations from a strict reading of "all texts" in guidelines:

| Element | Counted | Note |
|---|---|---|
| Chapters, headings, footnotes, tables | Yes | required by guidelines |
| Executive summary, COI/DA/keywords | Yes | "all texts" |
| Title page | Yes | `word-count` wraps `the_body`, so front matter is included |
| Acknowledgements | **No** | convention; not thesis research content |
| Bibliography | No | explicitly excluded by guidelines |
| Appendices | No | explicitly excluded by guidelines |
| Statement of authorship | No | legal declaration, not thesis content |
| Figure captions | Yes | required by guidelines |
| Tables & Table captions | Yes | required by guidelines |

## Files

```
main.typ          entry point
setup.typ         all document metadata and content fields
include/
  template.typ    show-rule functions (the_body, the_bibliography, the_appendix, the_statement)
  snippets.typ    helper functions
  papers.bib      main bibliography
  packages.bib    software/package references
```
