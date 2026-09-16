"""Render one PDF page in an isolated process."""

from __future__ import annotations

import argparse
import math
from pathlib import Path


def bounded_scale(
    width: float,
    height: float,
    resolution: int,
    max_image_dimension: int,
    max_image_pixels: int,
) -> float:
    """Return a PDF render scale constrained by configured image limits."""
    scale = resolution / 72
    pixel_width = max(1.0, width * scale)
    pixel_height = max(1.0, height * scale)
    dimension_factor = min(
        1.0,
        max_image_dimension / pixel_width,
        max_image_dimension / pixel_height,
    )
    pixel_factor = min(1.0, math.sqrt(max_image_pixels / (pixel_width * pixel_height)))
    return scale * min(dimension_factor, pixel_factor)


def render_pdf_page(
    pdf_path: str | Path,
    image_path: str | Path,
    resolution: int,
    max_image_dimension: int,
    max_image_pixels: int,
) -> None:
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
            width, height = page.get_size()
            scale = bounded_scale(
                width,
                height,
                resolution,
                max_image_dimension,
                max_image_pixels,
            )
            bitmap = page.render(scale=scale)
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
    parser.add_argument("max_image_dimension", type=int)
    parser.add_argument("max_image_pixels", type=int)
    args = parser.parse_args()
    render_pdf_page(
        args.pdf_path,
        args.image_path,
        args.resolution,
        args.max_image_dimension,
        args.max_image_pixels,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
