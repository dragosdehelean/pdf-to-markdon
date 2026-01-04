#!/usr/bin/env python3
"""
PDF → Markdown (max fidelity, single .md file)
- Text/layout: pymupdf4llm + PyMuPDF Layout (if installed)
- Tables: PyMuPDF Page.find_tables() → Markdown (NO images)

Usage:
  uv run python convert_pdf.py report.pdf
  uv run python convert_pdf.py report.pdf --chunks
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# IMPORTANT: import pymupdf.layout BEFORE pymupdf4llm so the layout engine is activated
try:
    import pymupdf.layout  # noqa: F401
    LAYOUT_AVAILABLE = True
except Exception:
    LAYOUT_AVAILABLE = False

import pymupdf
import pymupdf4llm


def _ensure_tessdata_prefix() -> None:
    """Helps PyMuPDF find traineddata if Tesseract is installed."""
    if os.environ.get("TESSDATA_PREFIX"):
        return
    try:
        tessdata = pymupdf.get_tessdata()
    except Exception:
        tessdata = None
    if tessdata:
        os.environ["TESSDATA_PREFIX"] = tessdata


def _p4llm_kwargs(*, page_chunks: bool, use_ocr: bool, header: bool, footer: bool) -> dict:
    # NOTE: Many knobs are ignored by PyMuPDF Layout; the important ones are
    # use_ocr / ocr_dpi / header / footer / force_text / write_images.
    # See PyMuPDF4LLM API docs for details.
    return {
        "page_chunks": page_chunks,
        "write_images": False,
        "embed_images": False,
        "force_text": True,
        "use_ocr": use_ocr,
        "ocr_dpi": 400,
        "header": header,
        "footer": footer,
        # Debug-friendly page boundaries (keeps page context in long reports)
        "page_separators": True,
        "show_progress": True,
        # For non-layout fallback, keep tiny footnotes too (ignored by Layout)
        "fontsize_limit": 0,
        # In Layout mode this is ignored, but it can matter in fallback mode
        "table_strategy": "lines_strict",
    }


def _dedupe_bboxes(bboxes: list[tuple[float, float, float, float]], bbox: tuple[float, float, float, float]) -> bool:
    """Return True if bbox is (almost) already in bboxes."""
    x0, y0, x1, y1 = bbox
    for bx0, by0, bx1, by1 in bboxes:
        if abs(x0 - bx0) < 2 and abs(y0 - by0) < 2 and abs(x1 - bx1) < 2 and abs(y1 - by1) < 2:
            return True
    return False


def extract_tables_markdown(
    doc: pymupdf.Document,
    *,
    page_numbers_0_based: list[int] | None = None,
) -> str:
    """Extract tables using PyMuPDF's native table detector.

    We try multiple strategies because some PDFs have clear grid lines ("lines"),
    others use background colors and benefit from "lines_strict", and OCRed pages
    may need "text" strategy.

    References:
    - Page.find_tables() strategies and parameters (PyMuPDF docs).
    - Table.to_markdown(clean, fill_empty) (PyMuPDF docs).
    """
    out: list[str] = []
    seen_bboxes: list[tuple[float, float, float, float]] = []

    strategies: list[tuple[str, dict]] = [
        ("lines_strict", {}),
        ("lines", {}),
        # "text" can be noisy; keep defaults but only accept non-trivial tables
        ("text", {"min_words_vertical": 3, "min_words_horizontal": 1}),
    ]

    pages_iter = page_numbers_0_based if page_numbers_0_based is not None else list(range(doc.page_count))

    for pno in pages_iter:
        page = doc[pno]
        page_tables: list[tuple[str, pymupdf.table.Table]] = []

        for strat, extra_kwargs in strategies:
            try:
                tf = page.find_tables(strategy=strat, **extra_kwargs)
            except Exception:
                continue

            for t in getattr(tf, "tables", []) or []:
                # Filter out tiny / degenerate "tables" (common with text strategy)
                if getattr(t, "row_count", 0) < 2 or getattr(t, "col_count", 0) < 2:
                    continue
                bbox = tuple(getattr(t, "bbox", (0, 0, 0, 0)))
                if _dedupe_bboxes(seen_bboxes, bbox):
                    continue
                seen_bboxes.append(bbox)
                page_tables.append((strat, t))

        if not page_tables:
            continue

        out.append(f"## Page {pno + 1}")
        for i, (strat, t) in enumerate(page_tables, start=1):
            out.append(
                f"### Table {i} (strategy={strat}, rows={t.row_count}, cols={t.col_count})"
            )
            try:
                out.append(t.to_markdown(clean=False, fill_empty=True).strip())
            except Exception as e:
                out.append(f"_Failed to export table to markdown: {e}_")
            out.append("")  # blank line

    if not out:
        return ""

    return "\n".join(["# Extracted tables (PyMuPDF Page.find_tables)", ""] + out).strip() + "\n"


def convert_pdf_to_markdown(
    pdf_path: str,
    output_path: str | None = None,
    *,
    use_ocr: bool = True,
    header: bool = True,
    footer: bool = True,
    include_tables: bool = True,
) -> str:
    """Convert a PDF into one Markdown file."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")
    if pdf_file.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF: {pdf_path}")

    print(f"📄 Processing: {pdf_file.name}")
    if LAYOUT_AVAILABLE:
        print("   ✓ PyMuPDF Layout active")
    else:
        print("   ⚠ PyMuPDF Layout not available (fallback mode)")

    _ensure_tessdata_prefix()

    # Open once and reuse for both text and table extraction.
    doc = pymupdf.open(pdf_file)

    md_text = pymupdf4llm.to_markdown(
        doc=doc,
        **_p4llm_kwargs(page_chunks=False, use_ocr=use_ocr, header=header, footer=footer),
    )

    if include_tables:
        tables_md = extract_tables_markdown(doc)
        if tables_md:
            md_text = md_text.rstrip() + "\n\n" + tables_md + "\n"

    md_file = Path(output_path) if output_path else pdf_file.with_suffix(".md")
    md_file.write_text(md_text, encoding="utf-8")
    print(f"✅ Saved: {md_file}")
    return str(md_file)


def convert_with_page_chunks(
    pdf_path: str,
    *,
    use_ocr: bool = True,
    header: bool = True,
    footer: bool = True,
    include_tables: bool = True,
) -> None:
    """Export one Markdown file per page (good for RAG chunking)."""
    pdf_file = Path(pdf_path)
    if not pdf_file.exists():
        raise FileNotFoundError(f"File not found: {pdf_path}")

    print(f"📄 Processing (page_chunks): {pdf_file.name}")
    if LAYOUT_AVAILABLE:
        print("   ✓ PyMuPDF Layout active")
    else:
        print("   ⚠ PyMuPDF Layout not available (fallback mode)")

    _ensure_tessdata_prefix()
    doc = pymupdf.open(pdf_file)

    pages_data = pymupdf4llm.to_markdown(
        doc=doc,
        **_p4llm_kwargs(page_chunks=True, use_ocr=use_ocr, header=header, footer=footer),
    )

    output_dir = pdf_file.parent / f"{pdf_file.stem}_pages"
    output_dir.mkdir(exist_ok=True)

    for i, page_dict in enumerate(pages_data):
        page_text = page_dict.get("text", "")
        if include_tables:
            page_tables = extract_tables_markdown(doc, page_numbers_0_based=[i]).strip()
            if page_tables:
                page_text = page_text.rstrip() + "\n\n" + page_tables + "\n"

        page_file = output_dir / f"page_{i + 1:03d}.md"
        page_file.write_text(page_text, encoding="utf-8")

    print(f"✅ Saved pages to: {output_dir}/")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert PDF reports to Markdown (max fidelity, no images).")

    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("--out", help="Output .md path (single-file mode)")
    parser.add_argument("--chunks", action="store_true", help="Write one .md file per page")

    parser.add_argument("--no-ocr", action="store_true", help="Disable OCR assist (Layout mode only)")
    parser.add_argument("--no-header", action="store_true", help="Drop repeating headers (Layout mode only)")
    parser.add_argument("--no-footer", action="store_true", help="Drop repeating footers (Layout mode only)")
    parser.add_argument("--no-tables", action="store_true", help="Skip explicit table extraction")

    args = parser.parse_args()

    use_ocr = not args.no_ocr
    header = not args.no_header
    footer = not args.no_footer
    include_tables = not args.no_tables

    try:
        if args.chunks:
            convert_with_page_chunks(
                args.pdf,
                use_ocr=use_ocr,
                header=header,
                footer=footer,
                include_tables=include_tables,
            )
        else:
            convert_pdf_to_markdown(
                args.pdf,
                output_path=args.out,
                use_ocr=use_ocr,
                header=header,
                footer=footer,
                include_tables=include_tables,
            )
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
