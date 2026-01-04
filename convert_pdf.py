#!/usr/bin/env python3
"""
Convertor PDF → Markdown folosind PyMuPDF4LLM + PyMuPDF-Layout
Utilizare: uv run python convert_pdf.py raport.pdf
"""

import sys
from pathlib import Path

# IMPORTANT: Importă pymupdf.layout ÎNAINTE de pymupdf4llm
# pentru a activa analiza îmbunătățită a layout-ului
try:
    import pymupdf.layout  # Activează PyMuPDF-Layout
    LAYOUT_AVAILABLE = True
except ImportError:
    LAYOUT_AVAILABLE = False

import pymupdf4llm


def convert_pdf_to_markdown(pdf_path: str, output_path: str | None = None) -> str:
    """
    Convertește un PDF în Markdown.
    
    Args:
        pdf_path: Calea către fișierul PDF
        output_path: Calea pentru fișierul .md (opțional, implicit: același nume ca PDF-ul)
    
    Returns:
        Calea către fișierul Markdown generat
    """
    pdf_file = Path(pdf_path)
    
    if not pdf_file.exists():
        raise FileNotFoundError(f"Fișierul nu există: {pdf_path}")
    
    if not pdf_file.suffix.lower() == ".pdf":
        raise ValueError(f"Fișierul nu este PDF: {pdf_path}")
    
    print(f"📄 Procesez: {pdf_file.name}")
    
    if LAYOUT_AVAILABLE:
        print("   ✓ PyMuPDF-Layout activ (analiză îmbunătățită)")
    else:
        print("   ⚠ PyMuPDF-Layout nu e instalat (rulează: uv add pymupdf-layout)")
    
    # Conversie PDF → Markdown / Config
    md_text = pymupdf4llm.to_markdown(
        doc=str(pdf_file),
        page_chunks=False,      # Un singur string pentru tot documentul
        write_images=False,     # Nu extrage imagini (pentru simplicitate)
    )
    
    # Determină calea output
    if output_path:
        md_file = Path(output_path)
    else:
        md_file = pdf_file.with_suffix(".md")
    
    # Salvează rezultatul
    md_file.write_text(md_text, encoding="utf-8")
    
    print(f"✅ Markdown salvat: {md_file}")
    print(f"   Dimensiune: {len(md_text):,} caractere")
    
    return str(md_file)


def convert_with_page_chunks(pdf_path: str) -> list[dict]:
    """
    Convertește PDF în Markdown cu metadata per pagină.
    Util pentru RAG/chunking.
    """
    pdf_file = Path(pdf_path)
    
    print(f"📄 Procesez cu page_chunks: {pdf_file.name}")
    
    if LAYOUT_AVAILABLE:
        print("   ✓ PyMuPDF-Layout activ (analiză îmbunătățită)")
    
    pages_data = pymupdf4llm.to_markdown(
        doc=str(pdf_file),
        page_chunks=True,  # Returnează listă de dict-uri per pagină
    )
    
    print(f"✅ Extras {len(pages_data)} pagini")
    
    # Salvează fiecare pagină separat
    output_dir = pdf_file.parent / f"{pdf_file.stem}_pages"
    output_dir.mkdir(exist_ok=True)
    
    for i, page in enumerate(pages_data):
        page_file = output_dir / f"page_{i+1:03d}.md"
        page_file.write_text(page["text"], encoding="utf-8")
    
    print(f"   Pagini salvate în: {output_dir}/")
    
    return pages_data


def main():
    if len(sys.argv) < 2:
        print("Utilizare:")
        print("  uv run python convert_pdf.py <fisier.pdf>")
        print("  uv run python convert_pdf.py <fisier.pdf> --chunks")
        print()
        print("Opțiuni:")
        print("  --chunks    Extrage fiecare pagină separat (util pentru RAG)")
        print()
        print(f"PyMuPDF-Layout: {'✓ Disponibil' if LAYOUT_AVAILABLE else '✗ Nu e instalat'}")
        sys.exit(1)
    
    pdf_path = sys.argv[1]
    use_chunks = "--chunks" in sys.argv
    
    try:
        if use_chunks:
            convert_with_page_chunks(pdf_path)
        else:
            convert_pdf_to_markdown(pdf_path)
    except FileNotFoundError as e:
        print(f"❌ Eroare: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Eroare la procesare: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()