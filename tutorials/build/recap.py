"""Build a tutorial's slide recap: selected pages from the lecture PDFs, in teaching order.

    make_recap("T03_cnns", [("slides/lectures/p3.pdf", [12, 13, 20]), ("slides/lectures/p4.pdf", [3, 7])])

Page numbers are 1-based, as shown in a PDF viewer. Output: tutorials/recap/<stem>_recap.pdf.
"""
from pathlib import Path
import pymupdf

REPO = Path(__file__).resolve().parents[2]


def make_recap(stem, selection):
    out = pymupdf.open()
    for pdf, pages in selection:
        src = pymupdf.open(REPO / pdf)
        for p in pages:
            out.insert_pdf(src, from_page=p - 1, to_page=p - 1)
    path = REPO / "tutorials" / "recap" / f"{stem}_recap.pdf"
    out.save(path, garbage=4, deflate=True)
    return path


def page_texts(pdf, max_chars=200):
    """Helper for choosing pages: [(page_no, first chars of text)]."""
    doc = pymupdf.open(REPO / pdf)
    return [(i + 1, " ".join(pg.get_text().split())[:max_chars]) for i, pg in enumerate(doc)]
