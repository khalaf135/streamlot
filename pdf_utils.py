import io
import fitz  # PyMuPDF

def parse_page_string(page_string: str, total_pages: int) -> list[int]:
    """Parse a string like '1-3, 5, 7' into a list of 0-indexed page numbers."""
    if not page_string or not page_string.strip():
        return list(range(total_pages))
        
    pages = set()
    parts = page_string.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            try:
                start_str, end_str = part.split("-", 1)
                start = int(start_str.strip()) - 1
                end = int(end_str.strip()) - 1
                start = max(0, start)
                end = min(total_pages - 1, end)
                if start <= end:
                    pages.update(range(start, end + 1))
            except ValueError:
                pass
        else:
            try:
                p = int(part) - 1
                if 0 <= p < total_pages:
                    pages.add(p)
            except ValueError:
                pass
                
    result = sorted(list(pages))
    if not result:
        return list(range(total_pages))
    return result

def trim_pdf_pages(pdf_bytes: bytes, page_string: str) -> bytes:
    """Return a new PDF byte string containing only the requested pages."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_to_keep = parse_page_string(page_string, len(doc))
    
    if len(pages_to_keep) == len(doc):
        doc.close()
        return pdf_bytes
        
    new_doc = fitz.open()
    for p in pages_to_keep:
        new_doc.insert_pdf(doc, from_page=p, to_page=p)
        
    out_bytes = new_doc.write()
    new_doc.close()
    doc.close()
    return out_bytes
