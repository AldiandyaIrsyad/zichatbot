"""Strip image-meta framing from VLM output.

A VLM asked to read a scanned page often narrates the *image* before
transcribing it — "Gambar ini adalah halaman ketiga dari sebuah dokumen resmi
berupa Keputusan Rektor…", followed by a "Deskripsi Umum Dokumen" section
heading. The framing is near-identical across every scanned decree in the
corpus, so it forms a dense cluster of generic text that matches almost any
question about an institutional document, crowding out the pages that actually
answer it.

This module removes the framing and keeps the transcription. Pure stdlib — no
infra imports, so ingestion and a backfill script can share it.
"""

from __future__ import annotations

import re
from typing import List

# The narration opener: "Gambar ini/tersebut/yang diberikan … adalah …".
# Anchored at the paragraph start so a mid-transcription mention of a figure
# ("Gambar 2 menunjukkan alur…") is never touched.
_META_OPENER = re.compile(
    r"^\s*(?:!\[[^\]]*\]\([^)]*\)\s*)?"          # an optional leading image tag
    r"(?:\*{0,2})"                                 # optional bold marker
    r"(?:Gambar|Citra|Halaman)\b[^.\n]{0,120}?"
    r"\b(?:adalah|merupakan|menunjukkan|memperlihatkan|berisi)\b",
    re.IGNORECASE,
)

# Standalone boilerplate section headings the narration introduces. The VLM
# often decorates them with an emoji ("## 📄 **Deskripsi Umum Dokumen**"), so
# anything that is not a word character, marker, or space is skipped between the
# hashes and the keyword.
_META_HEADING = re.compile(
    r"^\s*#{1,6}[\s\W]*?(?:Deskripsi|Analisis|Penjelasan|Ringkasan)"
    r"(?:\s+(?:Umum|Detail|Lengkap|Singkat))?"
    r"(?:\s+(?:Dokumen|Gambar|Halaman|Citra))?\s*\**\s*:?\s*$",
    re.IGNORECASE,
)

# A horizontal rule the narration uses to fence itself off from the content.
_RULE = re.compile(r"^\s*(?:-{3,}|\*{3,}|_{3,})\s*$")

# A connector line handing off from narration to transcription ("Berikut adalah
# deskripsi detail dari dokumen ini:"). Matched only as a whole line ending in
# a colon, so a sentence that merely starts this way is left alone.
_META_CONNECTOR = re.compile(
    r"^\s*\**\s*Berikut\s+(?:adalah\s+)?"
    r"(?:deskripsi|penjelasan|rincian|uraian|analisis|ringkasan)\b[^.\n]{0,80}:\s*\**\s*$",
    re.IGNORECASE,
)


# Emoji the VLM sprinkles through its output ("## 📄 **Deskripsi Umum**",
# "✅ **Dokumen ini sah**"). They are decoration the source document never had,
# they cost embedding tokens, and they leak into answers and citation badges.
#
# The ranges deliberately stop short of Mathematical Operators (U+2200-U+22FF),
# which is where "≤" and "≥" live — those are load-bearing in a tariff table
# ("Penghasilan ≤ Rp. 500.000"). Arrows and General Punctuation (en/em dashes,
# curly quotes) are likewise untouched.
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # pictographs, emoticons, transport, supplemental
    "\U0001F1E6-\U0001F1FF"   # regional indicators (flags)
    "☀-➿"           # misc symbols + dingbats (☑ ✅ ✍ ✂)
    "⬀-⯿"           # misc symbols and arrows (⭐ ⬅)
    "︎️"            # variation selectors (text/emoji presentation)
    "⃣"                  # combining enclosing keycap
    "‍"                  # zero-width joiner (multi-codepoint emoji)
    "]+"
)

# Swallow the whitespace either side of a removed run so "## 📄 **Judul**"
# collapses to "## **Judul**" rather than leaving a double space.
_EMOJI_RUN = re.compile(r"[ \t]*(?:" + _EMOJI.pattern + r")[ \t]*")


def strip_emoji(text: str) -> str:
    """Remove emoji decoration from VLM output, preserving legal typography.

    Whitespace around each removed run collapses to a single space, and lines
    reduced to nothing but decoration become empty rather than stray spaces.
    """
    if not text:
        return text

    cleaned_lines = []
    for line in text.splitlines():
        if _EMOJI.search(line):
            line = _EMOJI_RUN.sub(" ", line).strip()
        cleaned_lines.append(line)
    return "\n".join(cleaned_lines)


_FENCE_RE = re.compile(r"^[ \t]*```[a-zA-Z0-9_+-]*[ \t]*$", re.M)


def unwrap_markdown_table(text: str) -> str:
    """Drop code fences and any preamble above the first table row.

    VLMs return transcribed tables wrapped in a ```markdown fence and usually
    under a heading ("# Tarif Uang Kuliah Tunggal"). ``is_markdown_table``
    requires the text to *start* with "|", so either of those makes a real table
    read as opaque text: the chunker then stores the whole thing as one child
    instead of splitting it into row groups with the header repeated.

    Measured on a real UKT tariff page: 1 child of 6206 chars before, 37 children
    of ~1130 chars each (header in all 37) after — the difference between one
    diluted vector covering 39 study programmes and one chunk per programme.

    Only rewrites text that actually contains a table row; anything else is
    returned unchanged apart from fence removal.
    """
    without_fences = _FENCE_RE.sub("", text)
    lines = without_fences.splitlines()
    first_row = next(
        (i for i, line in enumerate(lines) if line.lstrip().startswith("|")), None
    )
    if first_row is None:
        return without_fences.strip()

    # A separator row must follow within the next two lines for this to be a
    # GFM table rather than a stray pipe in prose.
    window = lines[first_row : first_row + 3]
    if not any(
        line.strip().startswith("|") and all(c in "|:- " for c in line.strip())
        for line in window
    ):
        return without_fences.strip()

    return "\n".join(lines[first_row:]).strip()


def clean_vlm_output(text: str) -> str:
    """Full cleanup for one VLM extraction: drop the image narration, then the
    emoji decoration, then unwrap a transcribed table so the chunker can see it.
    The order matters — the narration's headings are matched while their emoji
    are still attached, which is how they were written, and table unwrapping
    runs last so it is not confused by narration above the table."""
    return unwrap_markdown_table(strip_emoji(strip_vlm_meta_description(text)))


def _is_blank(line: str) -> bool:
    return not line.strip()


def strip_vlm_meta_description(text: str) -> str:
    """Remove image-narration framing from one VLM page/figure extraction.

    Drops (a) a *leading* paragraph that narrates the image as an image, and
    (b) any standalone "Deskripsi … Dokumen" style heading, along with the
    horizontal rules and blank lines left behind. Everything else — the actual
    transcription, including tables and article text — is preserved verbatim.

    Conservative by construction: if the opener doesn't match at the very start
    of the text, nothing is removed. Returns the original text when stripping
    would leave nothing, so a page that is *only* narration still carries its
    (weak) content rather than becoming an empty chunk.
    """
    if not text or not text.strip():
        return text

    lines = text.splitlines()
    kept: List[str] = []
    index = 0

    # (a) The opening narration paragraph, only when it starts the text.
    if _META_OPENER.match(text.lstrip()):
        while index < len(lines) and _is_blank(lines[index]):
            index += 1
        while index < len(lines) and not _is_blank(lines[index]):
            index += 1

    # (b) Boilerplate headings/connectors and the rules/blanks fencing them.
    for line in lines[index:]:
        if _META_HEADING.match(line) or _META_CONNECTOR.match(line):
            # Drop the trailing rule/blank run this heading was sitting on.
            while kept and (_is_blank(kept[-1]) or _RULE.match(kept[-1])):
                kept.pop()
            continue
        kept.append(line)

    # Tidy the seam: leading rules/blanks left where the narration used to be.
    while kept and (_is_blank(kept[0]) or _RULE.match(kept[0])):
        kept.pop(0)
    while kept and _is_blank(kept[-1]):
        kept.pop()

    cleaned = "\n".join(kept).strip()
    return cleaned or text


def looks_like_vlm_meta_description(text: str) -> bool:
    """True if ``text`` opens with image narration or carries its boilerplate
    headings — the detector a backfill scan uses to pick documents to reingest.
    """
    if not text or not text.strip():
        return False
    if _META_OPENER.match(text.lstrip()):
        return True
    return any(
        _META_HEADING.match(line) or _META_CONNECTOR.match(line)
        for line in text.splitlines()
    )
