import io
from typing import Tuple, List
import httpx
from pypdf import PdfReader
from docx import Document as DocxDocument
from bs4 import BeautifulSoup
from eml_parser import EmlParser

CHUNK_SIZE = 1200
CHUNK_OVERLAP = 200


def _chunk_text(text: str) -> List[str]:
    chunks: List[str] = []
    start = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        chunk = text[start:end]
        chunks.append(chunk)
        if end == len(text):
            break
        start = end - CHUNK_OVERLAP
        if start < 0:
            start = 0
    return chunks


async def fetch_and_parse_document(url: str) -> Tuple[List[str], List[int]]:
    async with httpx.AsyncClient(timeout=60) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        content_type = resp.headers.get("content-type", "").lower()
        data = resp.content

    # Try by content-type first, fallback to sniffing bytes
    if "pdf" in content_type or data[:4] == b"%PDF":
        return _parse_pdf(data)
    if "word" in content_type or url.lower().endswith(".docx"):
        return _parse_docx(data)
    if "html" in content_type or url.lower().endswith((".htm", ".html")):
        return _parse_html(data)
    if "message/rfc822" in content_type or url.lower().endswith((".eml",)):
        return _parse_eml(data)

    # Fallback to treating as text
    text = data.decode("utf-8", errors="ignore")
    chunks = _chunk_text(text)
    return chunks, [None] * len(chunks)


def _parse_pdf(data: bytes) -> Tuple[List[str], List[int]]:
    reader = PdfReader(io.BytesIO(data))
    all_chunks: List[str] = []
    page_ids: List[int] = []
    for idx, page in enumerate(reader.pages):
        page_text = page.extract_text() or ""
        for chunk in _chunk_text(page_text):
            all_chunks.append(chunk)
            page_ids.append(idx + 1)
    return all_chunks, page_ids


def _parse_docx(data: bytes) -> Tuple[List[str], List[int]]:
    f = io.BytesIO(data)
    doc = DocxDocument(f)
    text = "\n".join(p.text for p in doc.paragraphs)
    chunks = _chunk_text(text)
    return chunks, [None] * len(chunks)


def _parse_html(data: bytes) -> Tuple[List[str], List[int]]:
    soup = BeautifulSoup(data, "lxml")
    text = soup.get_text(" ")
    chunks = _chunk_text(text)
    return chunks, [None] * len(chunks)


def _parse_eml(data: bytes) -> Tuple[List[str], List[int]]:
    parser = EmlParser(include_attachment_data=True, parse_attachments=True)
    eml = parser.decode_email_bytes(data)
    parts: List[str] = []
    subject = eml.get("header", {}).get("subject")
    if subject:
        parts.append(f"Subject: {subject}")
    body = eml.get("body", {})
    for key in ("plain", "html"):
        if key in body and body[key]:
            if isinstance(body[key], list):
                parts.extend(body[key])
            else:
                parts.append(body[key])
    text = "\n\n".join(parts)
    chunks = _chunk_text(text)
    return chunks, [None] * len(chunks)