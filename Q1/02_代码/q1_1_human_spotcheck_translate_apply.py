"""Create Chinese machine-translated copies of the Q1.1 reviewer workbooks.

Only blind_id/text are sent to Google Translate. Candidate/model columns and the
private sampling manifest are never sent. Existing reviewer workbooks are kept.
"""
from __future__ import annotations

import json
import re
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

from openpyxl import load_workbook


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "q1_1" / "human_spotcheck_final"
SOURCE_FILES = [OUT / "reviewer1.xlsx", OUT / "reviewer2.xlsx"]
DEST_FILES = [OUT / "reviewer1_中文机译版.xlsx", OUT / "reviewer2_中文机译版.xlsx"]
META_FILE = OUT / "translation_provenance.json"
MAX_CHARS = 2800
CODE_HEAVY_IDS = {"Q11-0792", "Q11-0353", "Q11-0089", "Q11-0529"}


def split_text(text: str, limit: int = MAX_CHARS) -> list[str]:
    """Split at paragraph/sentence boundaries where possible."""
    if not text:
        return [""]
    parts: list[str] = []
    rest = text
    while len(rest) > limit:
        cut = rest.rfind("\n", 0, limit)
        if cut < int(limit * 0.60):
            cut = rest.rfind(" ", 0, limit)
        if cut < int(limit * 0.60):
            cut = limit
        else:
            cut += 1
        parts.append(rest[:cut])
        rest = rest[cut:]
    if rest:
        parts.append(rest)
    return parts


def google_translate(text: str) -> str:
    query = urllib.parse.urlencode(
        {"client": "gtx", "sl": "auto", "tl": "zh-CN", "dt": "t", "q": text}
    )
    url = "https://translate.googleapis.com/translate_a/single?" + query
    last_error: Exception | None = None
    for attempt in range(5):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            with urllib.request.urlopen(req, timeout=45) as response:
                payload = json.loads(response.read().decode("utf-8"))
            chunks = payload[0]
            result = "".join(str(chunk[0]) for chunk in chunks if chunk and chunk[0])
            if not result:
                raise RuntimeError("Google Translate returned an empty translation")
            return result
        except Exception as exc:  # retry transient HTTP / parsing failures
            last_error = exc
            if attempt < 4:
                time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"Google Translate request failed: {last_error}")


def protect_technical_spans(text: str) -> tuple[str, dict[str, str]]:
    """Protect code, URLs, and math tokens from destructive translation."""
    saved: dict[str, str] = {}

    def hold(match: re.Match[str]) -> str:
        # Use symbols plus digits only: Google may translate words embedded in
        # otherwise uppercase placeholder strings, but preserves these tokens.
        token = f"@@{len(saved):06d}@@"
        saved[token] = match.group(0)
        return token

    # Whole fenced blocks, inline code, math expressions, URLs, and LaTeX
    # command names are syntax, not prose, and must survive unchanged.
    protected = re.sub(r"(?s)(`{3,}.*?`{3,}|~{3,}.*?~{3,})", hold, text)
    protected = re.sub(r"`[^`\n]+`", hold, protected)
    protected = re.sub(r"(?s)(\\\[.*?\\\]|\\\(.*?\\\)|\$\$.*?\$\$|\$[^$\n]+\$)", hold, protected)
    protected = re.sub(r"https?://[^\s)\]}>]+", hold, protected)
    # Keep escaped Unicode/hex sequences whole (e.g. the literal ``\\u200e``)
    # so a command-name placeholder cannot become adjacent to leftover digits.
    protected = re.sub(r"\\[A-Za-z]+[0-9]*", hold, protected)

    code_line = re.compile(
        r"^\s*(?:#include\b|import\b|from\b.+\bimport\b|namespace\b|using\b|"
        r"public\s*:|private\s*:|protected\s*:|class\s+\w|interface\s+\w|"
        r"typedef\b|function\b|async\b|await\b|const\b|let\b|var\b|return\b|"
        r"export\b|def\s+\w|SELECT\b|INSERT\b|UPDATE\b|DELETE\b|CREATE\b|"
        r"<\?php|\}\s*[,;]?\s*$|\{\s*$)"
    )
    out_lines = []
    for line in protected.splitlines(keepends=True):
        if "@@" not in line and code_line.search(line):
            out_lines.append(hold(re.match(r"(?s)(.*)", line)))
        else:
            out_lines.append(line)
    return "".join(out_lines), saved


def translate_text(blind_id: str, text: str) -> str:
    if blind_id in CODE_HEAVY_IDS:
        # These are source-code samples, not natural-language prose. Preserve
        # them exactly; translation would corrupt the evidence being rated.
        return text
    protected, saved = protect_technical_spans(text)
    translated = [google_translate(chunk) for chunk in split_text(protected)]
    # Translation responses sometimes trim a final line break; the sheet only
    # needs readable paragraphs, so preserve paragraph separation explicitly.
    result = "".join(translated)
    for token, original in saved.items():
        if token not in result:
            raise RuntimeError(f"technical placeholder was changed or lost: {token}")
        result = result.replace(token, original)
    return result


def read_review(path: Path):
    wb = load_workbook(path)
    ws = wb.active
    headers = [ws.cell(1, c).value for c in range(1, 8)]
    assert headers[0] == "blind_id" and headers[1] == "text", (path.name, headers)
    records = {}
    order = []
    for row in range(2, ws.max_row + 1):
        blind_id = ws.cell(row, 1).value
        if blind_id is None:
            continue
        blind_id = str(blind_id)
        assert blind_id not in records, f"duplicate blind_id in {path.name}: {blind_id}"
        text = ws.cell(row, 2).value
        assert isinstance(text, str) and text.strip(), f"missing text for {blind_id}"
        assert all(ws.cell(row, col).value in (None, "") for col in range(3, 8)), (
            f"ratings already entered; refusing to overwrite {path.name}:{blind_id}"
        )
        records[blind_id] = (row, text)
        order.append(blind_id)
    assert len(records) == 30, f"expected 30 samples in {path.name}; found {len(records)}"
    return wb, ws, records, order


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    r1, ws1, rec1, order1 = read_review(SOURCE_FILES[0])
    r2, ws2, rec2, order2 = read_review(SOURCE_FILES[1])
    assert set(rec1) == set(rec2), "reviewer workbooks do not contain the same sample IDs"
    for blind_id in rec1:
        assert rec1[blind_id][1] == rec2[blind_id][1], (
            f"source text mismatch between reviewer forms for {blind_id}"
        )

    translations: dict[str, str] = {}
    for i, blind_id in enumerate(order1, start=1):
        source = rec1[blind_id][1]
        translations[blind_id] = translate_text(blind_id, source)
        if not translations[blind_id].strip():
            raise RuntimeError(f"empty translation for {blind_id}")
        print(f"translated {i}/30: {blind_id} ({len(source)} chars)", flush=True)
        time.sleep(0.18)

    for source_path, dest_path, book, sheet, records in (
        (SOURCE_FILES[0], DEST_FILES[0], r1, ws1, rec1),
        (SOURCE_FILES[1], DEST_FILES[1], r2, ws2, rec2),
    ):
        for blind_id, (row, _source) in records.items():
            sheet.cell(row, 2).value = translations[blind_id]
        # Keep the machine-translation scope visible to raters while preserving
        # all columns and their positions for later score collection.
        sheet.cell(1, 1).value = "盲评编号"
        sheet.cell(1, 2).value = "评审文本（中文机译；代码/公式标记尽量保留）"
        book.save(dest_path)
        # Verify saved workbook retains the 30 rows, blank scores and Excel validations.
        verify = load_workbook(dest_path, data_only=False)
        vws = verify.active
        ids = [str(vws.cell(row, 1).value) for row in range(2, vws.max_row + 1)
               if vws.cell(row, 1).value is not None]
        assert len(ids) == 30 and len(set(ids)) == 30
        assert set(ids) == set(records)
        assert all(vws.cell(row, 2).value == translations[str(vws.cell(row, 1).value)]
                   for row in range(2, vws.max_row + 1) if vws.cell(row, 1).value is not None)
        assert all(vws.cell(row, col).value in (None, "")
                   for row in range(2, vws.max_row + 1) for col in range(3, 8))
        assert len(vws.data_validations.dataValidation) >= 1, "rating dropdown validations were lost"
        print(f"saved and verified: {dest_path}", flush=True)

    provenance = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "translation_provider": "Google Translate",
        "direction": "auto -> zh-CN",
        "sample_count": 30,
        "reviewer_forms": [p.name for p in DEST_FILES],
        "scope_note": (
            "These are machine-translated Chinese text aids. Ratings based on them are "
            "translation-assisted checks, not direct validation of the original-language text. "
            "Code-only rows remain original; marked code, URLs, and delimited math are protected. "
            "Other technical notation, names, and idioms may remain untranslated or be imperfect."
        ),
        "sample_ids_sent_to_provider": False,
        "candidate_scores_or_private_manifest_sent": False,
    }
    META_FILE.write_text(json.dumps(provenance, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"provenance: {META_FILE}", flush=True)


if __name__ == "__main__":
    main()
