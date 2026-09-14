"""Render one PDF page in an isolated process."""

from __future__ import annotations

import argparse
from pathlib import Path


def render_pdf_page(pdf_path: str | Path, image_path: str | Path, resolution: int) -> None:
    try:
        import pypdfium2
    except ImportError as exc:
        raise RuntimeError("PDF rendering requires pypdfium2") from exc

    destination = Path(image_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    document = pypdfium2.PdfDocument(str(pdf_path))
    try:
        page = document[0]
        try:
            bitmap = page.render(scale=max(1, int(resolution)) / 72)
            try:
                bitmap.to_pil().save(destination)
            finally:
                bitmap.close()
        finally:
            page.close()
    finally:
        document.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf_path")
    parser.add_argument("image_path")
    parser.add_argument("resolution", type=int)
    args = parser.parse_args()
    render_pdf_page(args.pdf_path, args.image_path, args.resolution)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
