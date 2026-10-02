# document_loader.py
# Reads PDFs, Word docs, Excel files, CSVs, and plain text files.
# Builds a searchable FAISS vector store with role-based access tagging.

import fitz          # PyMuPDF — PDF reading
import pytesseract
from PIL import Image
import os
import re
import csv
import json
from concurrent.futures import ThreadPoolExecutor
import hashlib
import docx          # python-docx — Word files
import openpyxl      # Excel files

from tqdm import tqdm
from langchain_classic.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_classic.schema import Document

# ── Windows Tesseract path ───────────────────────────────────────────────────
pytesseract.pytesseract.tesseract_cmd = r'C:/Program Files/Tesseract-OCR/tesseract.exe'

# ── Embedding model ──────────────────────────────────────────────────────────
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# ── Supported extensions ─────────────────────────────────────────────────────
SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".xlsx", ".xls", ".csv", ".txt"}

# Always resolve paths relative to THIS file's location, not the terminal folder.
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
VECTORSTORE = os.path.join(BASE_DIR, "vectorstore")


# ────────────────────────────────────────────────────────────────────────────
#  Text cleaning
# ────────────────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """
    Normalises text from any source — fixes hyphenated line breaks,
    collapses stray whitespace, removes blank lines.
    """
    text = re.sub(r'-\n(\w)', r'\1', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r' {2,}', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


# ────────────────────────────────────────────────────────────────────────────
#  Per-format extractors
# ────────────────────────────────────────────────────────────────────────────

def extract_text_from_pdf(path: str) -> list:
    """
    PDF extractor — native text extraction with OCR fallback for scanned pages.
    """
    doc = fitz.open(path)
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text("text")
        method = "native"

        if len(text.strip()) < 100:  # increased from 50 — skips near-blank pages faster
            pix = page.get_pixmap(dpi=300)
            img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
            text = pytesseract.image_to_string(img, lang='eng', config='--psm 1')
            method = "ocr"

        text = clean_text(text)

        if len(text.strip()) > 20:
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": os.path.basename(path),
                    "page": page_num + 1,
                    "extraction": method,
                    "file_type": "pdf"
                }
            })

        print(f"  Page {page_num + 1}/{len(doc)} [{method}]", end="\r")

    print()
    return pages


def extract_text_from_docx(path: str) -> list:
    """
    Word document extractor.
    Reads every paragraph AND every table cell.
    """
    doc = docx.Document(path)
    parts = []

    for para in doc.paragraphs:
        if para.text.strip():
            parts.append(para.text.strip())

    for table in doc.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells if cell.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    text = clean_text("\n\n".join(parts))

    if len(text.strip()) < 20:
        return []

    return [{
        "page_content": text,
        "metadata": {
            "source": os.path.basename(path),
            "page": 1,
            "extraction": "native",
            "file_type": "docx"
        }
    }]


def extract_text_from_excel(path: str) -> list:
    """
    Excel extractor — handles .xlsx and .xls files.
    Each worksheet becomes its own page.
    """
    wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
    pages = []

    for sheet_name in wb.sheetnames:
        ws = wb[sheet_name]
        rows = list(ws.iter_rows(values_only=True))

        if not rows:
            continue

        headers = [str(cell) if cell is not None else "" for cell in rows[0]]
        has_headers = any(h.strip() for h in headers)

        lines = []
        data_rows = rows[1:] if has_headers else rows

        for row in data_rows:
            if not any(cell is not None for cell in row):
                continue

            if has_headers:
                pairs = [
                    f"{headers[i]}: {cell}"
                    for i, cell in enumerate(row)
                    if cell is not None and str(cell).strip()
                    and i < len(headers)
                ]
                if pairs:
                    lines.append(" | ".join(pairs))
            else:
                cells = [str(c) for c in row if c is not None and str(c).strip()]
                if cells:
                    lines.append(" | ".join(cells))

        text = clean_text("\n".join(lines))

        if len(text.strip()) > 20:
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": os.path.basename(path),
                    "page": sheet_name,
                    "extraction": "native",
                    "file_type": "xlsx"
                }
            })

    wb.close()
    return pages


def extract_text_from_csv(path: str) -> list:
    """
    CSV extractor — same header:value formatting as Excel.
    """
    lines = []

    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, newline="", encoding=encoding) as f:
                reader = csv.reader(f)
                rows = list(reader)
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"  ⚠️  Could not decode {os.path.basename(path)} — skipping")
        return []

    if not rows:
        return []

    headers = rows[0]
    has_headers = any(h.strip() for h in headers)
    data_rows = rows[1:] if has_headers else rows

    for row in data_rows:
        if not any(cell.strip() for cell in row):
            continue

        if has_headers:
            pairs = [
                f"{headers[i]}: {cell}"
                for i, cell in enumerate(row)
                if cell.strip() and i < len(headers)
            ]
            if pairs:
                lines.append(" | ".join(pairs))
        else:
            cells = [c for c in row if c.strip()]
            if cells:
                lines.append(" | ".join(cells))

    text = clean_text("\n".join(lines))

    if len(text.strip()) < 20:
        return []

    return [{
        "page_content": text,
        "metadata": {
            "source": os.path.basename(path),
            "page": 1,
            "extraction": "native",
            "file_type": "csv"
        }
    }]


def extract_text_from_txt(path: str) -> list:
    """
    Plain text extractor — splits on double newlines into ~50-line pages.
    """
    for encoding in ("utf-8", "latin-1"):
        try:
            with open(path, encoding=encoding) as f:
                raw = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"  ⚠️  Could not decode {os.path.basename(path)} — skipping")
        return []

    blocks = [b.strip() for b in raw.split("\n\n") if b.strip()]
    pages = []
    chunk_blocks = []
    line_count = 0

    for block in blocks:
        chunk_blocks.append(block)
        line_count += block.count("\n") + 1
        if line_count >= 50:
            text = clean_text("\n\n".join(chunk_blocks))
            if len(text) > 20:
                pages.append({
                    "page_content": text,
                    "metadata": {
                        "source": os.path.basename(path),
                        "page": len(pages) + 1,
                        "extraction": "native",
                        "file_type": "txt"
                    }
                })
            chunk_blocks = []
            line_count = 0

    if chunk_blocks:
        text = clean_text("\n\n".join(chunk_blocks))
        if len(text) > 20:
            pages.append({
                "page_content": text,
                "metadata": {
                    "source": os.path.basename(path),
                    "page": len(pages) + 1,
                    "extraction": "native",
                    "file_type": "txt"
                }
            })

    return pages


# ────────────────────────────────────────────────────────────────────────────
#  Dispatcher
# ────────────────────────────────────────────────────────────────────────────

def extract_text_from_file(path: str) -> list:
    """
    Single entry point for all file types.
    """
    ext = os.path.splitext(path)[1].lower()

    extractors = {
        ".pdf":  extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".xlsx": extract_text_from_excel,
        ".xls":  extract_text_from_excel,
        ".csv":  extract_text_from_csv,
        ".txt":  extract_text_from_txt,
    }

    if ext not in extractors:
        print(f"  ⚠️  Unsupported format: {os.path.basename(path)} — skipping")
        return []

    try:
        return extractors[ext](path)
    except Exception as e:
        print(f"  ❌ Failed to read {os.path.basename(path)}: {e}")
        return []


# ────────────────────────────────────────────────────────────────────────────
#  Parallel file processor
# ────────────────────────────────────────────────────────────────────────────

def process_file(args: tuple) -> list:
    """
    Worker function for parallel processing.
    Takes a (filename, folder, access_level) tuple,
    extracts text, tags with access level, and returns pages.
    Used by ThreadPoolExecutor in build_vector_store.
    """
    filename, folder, access_level = args
    path = os.path.join(folder, filename)
    ext  = os.path.splitext(filename)[1].lower()
    print(f"\nLoading: {filename} [{access_level}] ({ext})")

    pages = extract_text_from_file(path)

    for page in pages:
        page["metadata"]["access_level"] = access_level

    return pages


# ────────────────────────────────────────────────────────────────────────────
#  Hash cache
# ────────────────────────────────────────────────────────────────────────────

def get_documents_hash() -> str:
    """
    MD5 fingerprint of every supported file across all three folders.
    """
    hasher = hashlib.md5()
    for folder in ["documents/public", "documents/employee", "documents/admin"]:
        if not os.path.exists(folder):
            continue
        for fname in sorted(os.listdir(folder)):
            ext = os.path.splitext(fname)[1].lower()
            if ext in SUPPORTED_EXTENSIONS:
                with open(os.path.join(folder, fname), "rb") as f:
                    hasher.update(f.read())
    return hasher.hexdigest()


# ────────────────────────────────────────────────────────────────────────────
#  Vector store
# ────────────────────────────────────────────────────────────────────────────

def load_vector_store():
    """Load an already-built vector store from disk."""
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    return FAISS.load_local(
        VECTORSTORE, embeddings,
        allow_dangerous_deserialization=True
    )


def build_vector_store():
    """
    Full pipeline:
    1. Check hash — skip rebuild if documents haven't changed
    2. Extract text from ALL supported file types in parallel
    3. Split into overlapping chunks
    4. Embed and save FAISS index
    """
    os.makedirs(VECTORSTORE, exist_ok=True)
    hash_file = os.path.join(VECTORSTORE, ".hash")
    current_hash = get_documents_hash()

    if os.path.exists(hash_file):
        with open(hash_file) as f:
            if f.read().strip() == current_hash:
                print("✓ Documents unchanged — loading existing vector store")
                return load_vector_store()

    folders = {
        "documents/public":   "public",
        "documents/employee": "employee",
        "documents/admin":    "admin"
    }

    # ── Build list of all files to process ──────────────────────────────────
    all_tasks = []
    for folder, access_level in folders.items():
        os.makedirs(folder, exist_ok=True)
        files = [
            f for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS
        ]
        for filename in files:
            all_tasks.append((filename, folder, access_level))

    if not all_tasks:
        print("⚠️  No supported files found — add files to the documents/ folders.")
        return None

    # ── Process files in parallel (4 workers) ───────────────────────────────
    # 4 workers is safe for most machines — increase to 8 if you have 8+ CPU cores.
    # PDFs with OCR are the slowest; non-PDF files finish almost instantly.
    raw_pages = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = executor.map(process_file, all_tasks)
        for pages in results:
            raw_pages.extend(pages)

    if not raw_pages:
        print("⚠️  No text extracted from any files.")
        return None

    # ── Chunk ────────────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " "]
    )

    docs = []
    for page in raw_pages:
        chunks = splitter.split_text(page["page_content"])
        for chunk in chunks:
            docs.append(Document(
                page_content=chunk,
                metadata=page["metadata"]
            ))

    print(f"\n✓ {len(docs)} chunks from {len(raw_pages)} pages/sheets across all files")

    # ── Embed ─────────────────────────────────────────────────────────────────
    print("Loading embedding model...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print("Building vector store...")
    vectorstore = None
    for i in tqdm(range(0, len(docs), 200), desc="Embedding chunks"):
        batch = docs[i:i + 200]
        if vectorstore is None:
            vectorstore = FAISS.from_documents(batch, embeddings)
        else:
            vectorstore.merge_from(FAISS.from_documents(batch, embeddings))

    vectorstore.save_local(VECTORSTORE)

    with open(hash_file, "w") as f:
        f.write(current_hash)

    print(f"✓ Vector store saved to {VECTORSTORE}")
    return vectorstore


def add_file_to_store(file_path: str, access_level: str, current_vectorstore):
    """
    Indexes a SINGLE new file and merges it into the existing vector store.
    Much faster than rebuilding everything — only processes the one new file.
    """
    print(f"\n[Incremental index] Adding: {os.path.basename(file_path)} [{access_level}]")

    pages = extract_text_from_file(file_path)

    if not pages:
        print(f"  ⚠️  No text extracted from {os.path.basename(file_path)} — skipping")
        return current_vectorstore

    for page in pages:
        page["metadata"]["access_level"] = access_level

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=400,
        separators=["\n\n", "\n", ". ", " "]
    )

    docs = []
    for page in pages:
        for chunk in splitter.split_text(page["page_content"]):
            docs.append(Document(page_content=chunk, metadata=page["metadata"]))

    print(f"  ✓ {len(docs)} chunks from {len(pages)} pages")

    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    new_store = FAISS.from_documents(docs, embeddings)

    if current_vectorstore is None:
        current_vectorstore = new_store
    else:
        current_vectorstore.merge_from(new_store)

    current_vectorstore.save_local(VECTORSTORE)

    with open(os.path.join(VECTORSTORE, ".hash"), "w") as f:
        f.write(get_documents_hash())

    print(f"  ✓ Store updated — {os.path.basename(file_path)} is now searchable")
    return current_vectorstore