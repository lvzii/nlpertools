#!/usr/bin/env python3
"""Convert a PDF file into Markdown text.

Usage:
    python pdf_to_markdown.py input.pdf -o output.md

Dependency:
    pip install pymupdf
"""

from __future__ import annotations

import argparse
import re
import statistics
from pathlib import Path

import fitz  # PyMuPDF


LIST_PATTERN = re.compile(r"^(([-*])|(\d+[.)]))\s+")


def normalize_whitespace(text: str) -> str:
    """Collapse duplicated spaces while keeping simple readability."""
    return re.sub(r"\s+", " ", text).strip()


def heading_level(font_size: float, body_size: float, is_bold: bool) -> int | None:
    """Infer heading level from font size relative to body text size."""
    ratio = font_size / body_size if body_size else 1.0
    if ratio >= 1.60:
        return 1
    if ratio >= 1.35:
        return 2
    if ratio >= 1.18 and is_bold:
        return 3
    return None


def extract_lines_with_style(page: fitz.Page) -> list[tuple[str, float, bool]]:
    """Return lines as (text, max_font_size, any_bold)."""
    raw = page.get_text("dict")
    result: list[tuple[str, float, bool]] = []

    for block in raw.get("blocks", []):
        if block.get("type") != 0:
            continue

        for line in block.get("lines", []):
            spans = line.get("spans", [])
            if not spans:
                continue

            text = "".join(span.get("text", "") for span in spans)
            text = normalize_whitespace(text)
            if not text:
                continue

            max_size = max(float(span.get("size", 0.0)) for span in spans)
            any_bold = any(int(span.get("flags", 0)) & 16 for span in spans)
            result.append((text, max_size, any_bold))

    return result


def estimate_body_font_size(document: fitz.Document) -> float:
    """Estimate the dominant body font size across the document."""
    sizes: list[float] = []
    for page in document:
        raw = page.get_text("dict")
        for block in raw.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    txt = normalize_whitespace(span.get("text", ""))
                    if txt:
                        sizes.append(float(span.get("size", 0.0)))

    if not sizes:
        return 12.0

    try:
        return statistics.mode(round(size, 1) for size in sizes)
    except statistics.StatisticsError:
        return statistics.median(sizes)


def convert_pdf_to_markdown(pdf_path: Path, start_page: int, end_page: int | None) -> str:
    """Convert selected pages of a PDF to Markdown string."""
    doc = fitz.open(pdf_path)
    try:
        body_size = estimate_body_font_size(doc)
        chunks: list[str] = []

        first_page_index = max(start_page - 1, 0)
        last_page_index = (end_page - 1) if end_page else (len(doc) - 1)
        last_page_index = min(last_page_index, len(doc) - 1)

        for page_index in range(first_page_index, last_page_index + 1):
            page = doc[page_index]
            lines = extract_lines_with_style(page)
            if not lines:
                continue

            chunks.append(f"\n<!-- Page {page_index + 1} -->\n")

            for text, font_size, is_bold in lines:
                if LIST_PATTERN.match(text):
                    chunks.append(text)
                    continue

                level = heading_level(font_size, body_size, is_bold)
                if level:
                    chunks.append(f"{'#' * level} {text}")
                else:
                    chunks.append(text)

            chunks.append("")

        return "\n".join(chunks).strip() + "\n"
    finally:
        doc.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert PDF to Markdown")
    parser.add_argument("input", type=Path, help="Input PDF path")
    parser.add_argument("-o", "--output", type=Path, help="Output Markdown path")
    parser.add_argument("--start-page", type=int, default=1, help="1-based start page")
    parser.add_argument("--end-page", type=int, default=None, help="1-based end page")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.input.exists():
        raise SystemExit(f"Input file not found: {args.input}")

    if args.input.suffix.lower() != ".pdf":
        raise SystemExit("Input file must be a .pdf")

    output_path = args.output or args.input.with_suffix(".md")
    markdown = convert_pdf_to_markdown(args.input, args.start_page, args.end_page)
    output_path.write_text(markdown, encoding="utf-8")
    print(f"Markdown written to: {output_path}")


if __name__ == "__main__":
    main()
