import io
import os
import re
import json
import hashlib
import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse, parse_qs

import numpy as np
import streamlit as st
import faiss
import requests
from bs4 import BeautifulSoup
from docx import Document
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from groq import Groq


# -----------------------------
# App configuration
# -----------------------------
st.set_page_config(
    page_title="Advanced RAG Document Assistant",
    page_icon="📚",
    layout="wide",
)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md"}
DEFAULT_CHUNK_SIZE = 900
DEFAULT_CHUNK_OVERLAP = 150
DEFAULT_TOP_K = 5
DEFAULT_KEYWORD_WEIGHT = 0.30
DEFAULT_SEMANTIC_WEIGHT = 0.70


# -----------------------------
# Utility functions
# -----------------------------
def normalize_text(text: str) -> str:
    """Normalize whitespace while preserving paragraph boundaries reasonably."""
    text = text.replace("\x00", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def extension_from_name(name: str) -> str:
    return Path(name).suffix.lower()


def format_page(page: Any) -> str:
    if page in (None, "", "N/A"):
        return "N/A"
    return str(page)


# -----------------------------
# Document extraction
# -----------------------------
def extract_pdf(data: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract PDF text page-by-page and preserve page numbers."""
    reader = PdfReader(io.BytesIO(data))
    pages = []
    all_text = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = normalize_text(page.extract_text() or "")
        if text:
            pages.append({
                "filename": filename,
                "page": page_number,
                "text": text,
            })
            all_text.append(text)

    return "\n\n".join(all_text), pages


def extract_docx(data: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract DOCX paragraphs. DOCX has no reliable page metadata in standard python-docx."""
    doc = Document(io.BytesIO(data))
    paragraphs = []

    for paragraph in doc.paragraphs:
        text = normalize_text(paragraph.text)
        if text:
            paragraphs.append(text)

    full_text = "\n\n".join(paragraphs)
    pages = [{
        "filename": filename,
        "page": None,
        "text": full_text,
    }] if full_text else []

    return full_text, pages


def extract_txt(data: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract UTF-8 text with fallback decoding."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    text = normalize_text(text)
    pages = [{
        "filename": filename,
        "page": None,
        "text": text,
    }] if text else []

    return text, pages


def extract_md(data: bytes, filename: str) -> Tuple[str, List[Dict[str, Any]]]:
    """Extract Markdown as text while preserving headings/content."""
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError:
        text = data.decode("latin-1", errors="replace")

    text = normalize_text(text)
    pages = [{
        "filename": filename,
        "page": None,
        "text": text,
    }] if text else []

    return text, pages


def extract_document(data: bytes, filename: str) -> Dict[str, Any]:
    """Route a supported document to its dedicated extraction function."""
    ext = extension_from_name(filename)

    if ext == ".pdf":
        text, units = extract_pdf(data, filename)
    elif ext == ".docx":
        text, units = extract_docx(data, filename)
    elif ext == ".txt":
        text, units = extract_txt(data, filename)
    elif ext == ".md":
        text, units = extract_md(data, filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return {
        "filename": filename,
        "extension": ext,
        "text": text,
        "units": units,
        "characters": len(text),
        "pages_or_units": len(units),
        "file_hash": file_hash(data),
    }


# -----------------------------
# Chunking
# -----------------------------
def split_words_with_metadata(
    text: str,
    filename: str,
    page: Any,
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """Create overlapping word-window chunks."""
    words = text.split()
    if not words:
        return []

    if chunk_overlap >= chunk_size:
        raise ValueError("Chunk overlap must be smaller than chunk size.")

    chunks = []
    start = 0
    chunk_id = 0
    step = chunk_size - chunk_overlap

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk_text = " ".join(words[start:end]).strip()

        if chunk_text:
            chunks.append({
                "chunk_id": chunk_id,
                "filename": filename,
                "page": page,
                "text": chunk_text,
                "word_start": start,
                "word_end": end,
            })
            chunk_id += 1

        if end >= len(words):
            break

        start += step

    return chunks


def create_chunks(
    extracted_documents: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
) -> List[Dict[str, Any]]:
    """Chunk extracted document units while preserving filename/page metadata."""
    all_chunks = []

    for document in extracted_documents:
        for unit in document["units"]:
            chunks = split_words_with_metadata(
                unit["text"],
                unit["filename"],
                unit.get("page"),
                chunk_size,
                chunk_overlap,
            )
            all_chunks.extend(chunks)

    for idx, chunk in enumerate(all_chunks):
        chunk["global_chunk_id"] = idx

    return all_chunks


# -----------------------------
# Embeddings
# -----------------------------
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache the Sentence Transformers model."""
    return SentenceTransformer(model_name)


def embed_texts(
    texts: List[str],
    model: SentenceTransformer,
    batch_size: int = 32,
) -> np.ndarray:
    """Create normalized sentence-transformer embeddings."""
    if not texts:
        return np.empty((0, model.get_sentence_embedding_dimension()), dtype="float32")

    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=False,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )
    return np.asarray(embeddings, dtype="float32")


def build_faiss_index(embeddings: np.ndarray):
    """Build an inner-product FAISS index for normalized embeddings."""
    if embeddings.size == 0:
        return None

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings.astype("float32"))
    return index


# -----------------------------
# Keyword search
# -----------------------------
STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "than", "to", "of",
    "in", "on", "for", "from", "with", "by", "is", "are", "was", "were",
    "be", "been", "being", "this", "that", "these", "those", "it", "its",
    "as", "at", "into", "about", "what", "which", "who", "when", "where",
    "why", "how", "can", "could", "would", "should", "do", "does", "did",
    "i", "we", "you", "they", "he", "she", "them", "my", "your", "our",
    "their", "me", "please", "tell", "give", "show", "provide",
}


def tokenize(text: str) -> List[str]:
    return re.findall(r"\b[a-zA-Z0-9][a-zA-Z0-9_-]{1,}\b", text.lower())


def important_keywords(query: str) -> List[str]:
    tokens = tokenize(query)
    return [token for token in tokens if token not in STOPWORDS]


def keyword_score(query: str, text: str) -> float:
    """Score overlap of important query words with a chunk."""
    query_terms = set(important_keywords(query))
    if not query_terms:
        return 0.0

    chunk_terms = set(tokenize(text))
    overlap = query_terms.intersection(chunk_terms)
    return len(overlap) / len(query_terms)


def keyword_search(query: str, chunks: List[Dict[str, Any]], top_k: int):
    scored = []
    for chunk in chunks:
        score = keyword_score(query, chunk["text"])
        scored.append((score, chunk))

    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:top_k]


# -----------------------------
# Hybrid search
# -----------------------------
def semantic_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index,
    model: SentenceTransformer,
    top_k: int,
):
    if not chunks or index is None:
        return []

    query_embedding = embed_texts([query], model)
    scores, indices = index.search(query_embedding, min(top_k, len(chunks)))

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0:
            continue
        results.append((float(score), chunks[int(idx)]))

    return results


def hybrid_search(
    query: str,
    chunks: List[Dict[str, Any]],
    embeddings: np.ndarray,
    index,
    model: SentenceTransformer,
    top_k: int = DEFAULT_TOP_K,
    semantic_weight: float = DEFAULT_SEMANTIC_WEIGHT,
    keyword_weight: float = DEFAULT_KEYWORD_WEIGHT,
) -> List[Dict[str, Any]]:
    """Combine semantic and keyword scores and return ranked chunks."""
    if not chunks:
        return []

    candidate_k = min(max(top_k * 4, 10), len(chunks))

    semantic_results = semantic_search(
        query, chunks, embeddings, index, model, candidate_k
    )
    keyword_results = keyword_search(query, chunks, candidate_k)

    combined: Dict[int, Dict[str, Any]] = {}

    # Semantic scores from normalized embeddings are normally in [-1, 1].
    # Clamp to [0, 1] for a stable weighted ranking.
    for score, chunk in semantic_results:
        cid = chunk["global_chunk_id"]
        combined.setdefault(cid, {"chunk": chunk, "semantic_score": 0.0, "keyword_score": 0.0})
        combined[cid]["semantic_score"] = max(0.0, min(1.0, (score + 1.0) / 2.0))

    for score, chunk in keyword_results:
        cid = chunk["global_chunk_id"]
        combined.setdefault(cid, {"chunk": chunk, "semantic_score": 0.0, "keyword_score": 0.0})
        combined[cid]["keyword_score"] = score

    ranked = []
    for item in combined.values():
        final_score = (
            semantic_weight * item["semantic_score"]
            + keyword_weight * item["keyword_score"]
        )

        chunk = dict(item["chunk"])
        chunk["semantic_score"] = item["semantic_score"]
        chunk["keyword_score"] = item["keyword_score"]
        chunk["hybrid_score"] = final_score
        ranked.append(chunk)

    ranked.sort(key=lambda x: x["hybrid_score"], reverse=True)
    return ranked[:top_k]


# -----------------------------
# Google Drive public/shared-link ingestion
# -----------------------------
def extract_drive_file_id(url: str) -> str | None:
    """Extract a Google Drive file ID from common share URL formats."""
    patterns = [
        r"/file/d/([a-zA-Z0-9_-]+)",
        r"/document/d/([a-zA-Z0-9_-]+)",
        r"/spreadsheets/d/([a-zA-Z0-9_-]+)",
        r"/presentation/d/([a-zA-Z0-9_-]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)

    parsed = urlparse(url)
    query = parse_qs(parsed.query)
    if "id" in query and query["id"]:
        return query["id"][0]

    return None


def download_drive_file(file_id: str) -> Tuple[bytes, str]:
    """Download a publicly accessible Google Drive file."""
    url = f"https://drive.google.com/uc?export=download&id={file_id}"
    response = requests.get(
        url,
        timeout=60,
        allow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0"},
    )
    response.raise_for_status()

    content_type = response.headers.get("content-type", "").lower()

    # Large Drive files may return an HTML confirmation page.
    if "text/html" in content_type:
        soup = BeautifulSoup(response.text, "html.parser")

        # Google sometimes includes a confirmation form.
        form = soup.find("form")
        if form:
            action = form.get("action") or url
            params = {}
            for inp in form.find_all("input"):
                name = inp.get("name")
                value = inp.get("value", "")
                if name:
                    params[name] = value

            response = requests.get(
                action,
                params=params,
                timeout=60,
                allow_redirects=True,
                headers={"User-Agent": "Mozilla/5.0"},
            )
            response.raise_for_status()

    data = response.content
    return data, response.headers.get("content-type", "")


def infer_drive_filename(url: str, content_type: str) -> str:
    """Infer a reasonable filename when Drive doesn't expose one."""
    parsed = urlparse(url)
    path_name = Path(parsed.path).name

    if path_name and "." in path_name:
        return path_name

    guessed = mimetypes.guess_extension(content_type.split(";")[0].strip())
    return f"google_drive_document{guessed or '.txt'}"


def load_google_drive_link(url: str) -> List[Dict[str, Any]]:
    """
    Load a public/shared Google Drive file.

    Folder links require Drive API/OAuth to enumerate arbitrary private folder
    contents. This implementation supports direct public/shared file links and
    explains that limitation in the UI.
    """
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise ValueError("Could not find a Google Drive file ID in the supplied link.")

    data, content_type = download_drive_file(file_id)
    filename = infer_drive_filename(url, content_type)

    # Try to infer extension from Drive's response headers.
    ext = extension_from_name(filename)
    if ext not in SUPPORTED_EXTENSIONS:
        # Google export/download responses can have generic MIME types.
        mime_map = {
            "application/pdf": ".pdf",
            "application/vnd.openxmlformats-officedocument.wordprocessingml.document": ".docx",
            "text/plain": ".txt",
            "text/markdown": ".md",
        }
        ext = mime_map.get(content_type.split(";")[0].strip(), ext)
        if ext in SUPPORTED_EXTENSIONS:
            filename = Path(filename).stem + ext

    if extension_from_name(filename) not in SUPPORTED_EXTENSIONS:
        raise ValueError(
            "The Drive file type could not be identified as PDF, DOCX, TXT, or MD. "
            "Rename/provide a supported direct file link or use local upload."
        )

    return [{
        "filename": filename,
        "data": data,
        "source": "Google Drive",
        "source_url": url,
        "file_hash": file_hash(data),
    }]


# -----------------------------
# Processing pipeline
# -----------------------------
def process_source_files(
    source_files: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
):
    """Extract, chunk, embed, and index documents once."""
    extracted = []

    for source in source_files:
        try:
            document = extract_document(source["data"], source["filename"])
            document["source"] = source.get("source", "Local Upload")
            document["source_url"] = source.get("source_url")
            extracted.append(document)
        except Exception as exc:
            st.error(f"Could not process {source['filename']}: {exc}")

    chunks = create_chunks(extracted, chunk_size, chunk_overlap)

    model = load_embedding_model(model_name)
    texts = [chunk["text"] for chunk in chunks]
    embeddings = embed_texts(texts, model)
    index = build_faiss_index(embeddings)

    return extracted, chunks, embeddings, index


def make_processing_signature(
    source_files: List[Dict[str, Any]],
    chunk_size: int,
    chunk_overlap: int,
    model_name: str,
) -> str:
    payload = {
        "files": sorted(
            [(f["filename"], f["file_hash"]) for f in source_files]
        ),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "model_name": model_name,
    }
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


# -----------------------------
# Groq
# -----------------------------
def get_groq_api_key() -> str | None:
    """Read Groq API key from Streamlit secrets or environment."""
    try:
        secret_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        secret_key = None

    return secret_key or os.getenv("GROQ_API_KEY")


@st.cache_resource
def get_groq_client(api_key: str):
    return Groq(api_key=api_key)


def answer_with_groq(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    model_name: str,
) -> str:
    api_key = get_groq_api_key()
    if not api_key:
        raise RuntimeError(
            "GROQ_API_KEY is missing. Add it to Streamlit Secrets or your environment."
        )

    client = get_groq_client(api_key)

    context_parts = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        page = format_page(chunk.get("page"))
        context_parts.append(
            f"[SOURCE {i}]\n"
            f"Filename: {chunk['filename']}\n"
            f"Page: {page}\n"
            f"Content:\n{chunk['text']}"
        )

    context = "\n\n".join(context_parts)

    system_prompt = """You are an AI Document Assistant using retrieval-augmented generation.

Answer the user's question ONLY from the supplied document context.

Rules:
1. Do not use outside knowledge, assumptions, or invented facts.
2. If the answer is not supported by the supplied context, say:
   "The requested information is not available in the provided documents."
3. Be precise and concise.
4. When useful, mention the relevant filename and page number from the supplied context.
5. Do not claim that a source says something unless it is present in the supplied context.
"""

    user_prompt = f"""DOCUMENT CONTEXT:
{context}

USER QUESTION:
{question}

Answer using only the document context above."""

    completion = client.chat.completions.create(
        model=model_name,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    )

    return completion.choices[0].message.content.strip()


# -----------------------------
# Session state
# -----------------------------
def initialize_state():
    defaults = {
        "source_files": [],
        "extracted_documents": [],
        "chunks": [],
        "embeddings": None,
        "faiss_index": None,
        "processing_signature": None,
        "chat_history": [],
        "last_retrieved": [],
    }

    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


initialize_state()


# -----------------------------
# Sidebar
# -----------------------------
st.title("📚 Advanced RAG Document Assistant")
st.caption(
    "Hybrid Retrieval-Augmented Generation using Sentence Transformers, FAISS, "
    "keyword search, and Groq."
)

with st.sidebar:
    st.header("⚙️ RAG Settings")

    chunk_size = st.slider(
        "Chunk size (words)",
        min_value=300,
        max_value=1800,
        value=DEFAULT_CHUNK_SIZE,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk overlap (words)",
        min_value=0,
        max_value=400,
        value=DEFAULT_CHUNK_OVERLAP,
        step=25,
    )

    top_k = st.slider(
        "Retrieved chunks",
        min_value=1,
        max_value=10,
        value=DEFAULT_TOP_K,
    )

    semantic_weight = st.slider(
        "Semantic weight",
        min_value=0.0,
        max_value=1.0,
        value=DEFAULT_SEMANTIC_WEIGHT,
        step=0.05,
    )
    keyword_weight = 1.0 - semantic_weight

    embedding_model_name = st.selectbox(
        "Embedding model",
        [
            "sentence-transformers/all-MiniLM-L6-v2",
            "sentence-transformers/all-mpnet-base-v2",
        ],
        index=0,
    )

    groq_model = st.text_input(
        "Groq model",
        value="openai/gpt-oss-20b",
        help="Use a currently available Groq chat model.",
    )

    st.divider()
    st.caption(f"Semantic weight: {semantic_weight:.2f}")
    st.caption(f"Keyword weight: {keyword_weight:.2f}")


# -----------------------------
# Sources
# -----------------------------
tab_local, tab_drive, tab_data, tab_embeddings = st.tabs(
    ["📤 Local Documents", "☁️ Google Drive", "📊 Processed Data", "🧬 Embeddings"]
)

with tab_local:
    uploaded_files = st.file_uploader(
        "Upload PDF, DOCX, TXT, or MD files",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if uploaded_files:
        new_files = []
        for uploaded in uploaded_files:
            data = uploaded.getvalue()
            new_files.append({
                "filename": uploaded.name,
                "data": data,
                "source": "Local Upload",
                "source_url": None,
                "file_hash": file_hash(data),
            })

        # Merge by content hash so re-runs do not duplicate documents.
        merged = {
            item["file_hash"]: item
            for item in st.session_state.source_files
        }
        for item in new_files:
            merged[item["file_hash"]] = item

        st.session_state.source_files = list(merged.values())

    st.info(
        f"{len(st.session_state.source_files)} unique document(s) currently selected."
    )

with tab_drive:
    st.subheader("Google Drive source")
    drive_url = st.text_input(
        "Paste a public/shared Google Drive file link",
        placeholder="https://drive.google.com/file/d/...",
    )

    if st.button("➕ Add Google Drive File", use_container_width=True):
        if not drive_url.strip():
            st.warning("Please paste a Google Drive link.")
        else:
            try:
                drive_files = load_google_drive_link(drive_url.strip())
                existing_hashes = {
                    item["file_hash"] for item in st.session_state.source_files
                }

                added = 0
                for item in drive_files:
                    if item["file_hash"] not in existing_hashes:
                        st.session_state.source_files.append(item)
                        added += 1

                st.success(f"Added {added} new Drive file(s).")
            except Exception as exc:
                st.error(f"Google Drive loading failed: {exc}")

    st.caption(
        "The no-OAuth implementation supports directly downloadable public/shared "
        "Drive files. Arbitrary private folder enumeration requires Google Drive "
        "API/OAuth credentials."
    )


# -----------------------------
# Process documents
# -----------------------------
source_signature = make_processing_signature(
    st.session_state.source_files,
    chunk_size,
    chunk_overlap,
    embedding_model_name,
)

needs_processing = (
    bool(st.session_state.source_files)
    and source_signature != st.session_state.processing_signature
)

if needs_processing:
    st.warning(
        "Document settings or sources changed. Click **Process / Rebuild Index** "
        "to update the RAG index."
    )

process_col1, process_col2 = st.columns([3, 1])

with process_col1:
    if st.button(
        "🚀 Process / Rebuild RAG Index",
        type="primary",
        use_container_width=True,
        disabled=not bool(st.session_state.source_files),
    ):
        with st.spinner("Extracting, chunking, embedding, and indexing documents..."):
            (
                extracted,
                chunks,
                embeddings,
                index,
            ) = process_source_files(
                st.session_state.source_files,
                chunk_size,
                chunk_overlap,
                embedding_model_name,
            )

            st.session_state.extracted_documents = extracted
            st.session_state.chunks = chunks
            st.session_state.embeddings = embeddings
            st.session_state.faiss_index = index
            st.session_state.processing_signature = source_signature
            st.session_state.chat_history = []
            st.session_state.last_retrieved = []

        st.success(
            f"Processed {len(extracted)} document(s) and created {len(chunks)} chunks."
        )

with process_col2:
    if st.button("🗑️ Clear", use_container_width=True):
        for key in [
            "source_files",
            "extracted_documents",
            "chunks",
            "embeddings",
            "faiss_index",
            "processing_signature",
            "chat_history",
            "last_retrieved",
        ]:
            st.session_state[key] = [] if key in {
                "source_files",
                "extracted_documents",
                "chunks",
                "chat_history",
                "last_retrieved",
            } else None
        st.rerun()


# -----------------------------
# Processed information
# -----------------------------
with tab_data:
    st.subheader("Extracted document information")

    docs = st.session_state.extracted_documents

    if not docs:
        st.info("No processed documents yet.")
    else:
        for doc in docs:
            with st.expander(f"📄 {doc['filename']}"):
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Characters", f"{doc['characters']:,}")
                c2.metric("Pages/units", doc["pages_or_units"])
                c3.metric("Type", doc["extension"].upper())
                c4.metric("Source", doc["source"])

                st.text_area(
                    "Extracted text preview",
                    doc["text"][:5000],
                    height=220,
                    key=f"preview_{doc['file_hash']}",
                )

    st.subheader("Chunk statistics")
    if st.session_state.chunks:
        st.metric("Created chunks", len(st.session_state.chunks))

        chunk_rows = []
        for chunk in st.session_state.chunks[:100]:
            chunk_rows.append({
                "Chunk ID": chunk["global_chunk_id"],
                "Filename": chunk["filename"],
                "Page": format_page(chunk["page"]),
                "Words": len(chunk["text"].split()),
                "Preview": chunk["text"][:180],
            })

        st.dataframe(chunk_rows, use_container_width=True)
        if len(st.session_state.chunks) > 100:
            st.caption("Showing the first 100 chunks.")


# -----------------------------
# Embedding display
# -----------------------------
with tab_embeddings:
    st.subheader("Sentence Transformer embeddings")

    embeddings = st.session_state.embeddings
    chunks = st.session_state.chunks

    if embeddings is None or not chunks:
        st.info("Process documents to generate reusable embeddings.")
    else:
        st.write(
            f"Embedding matrix shape: **{embeddings.shape[0]} × {embeddings.shape[1]}**"
        )
        st.caption(
            "Embeddings are generated once during document processing and stored in "
            "Streamlit session state. Questions only create a query embedding."
        )

        rows = []
        for i, chunk in enumerate(chunks[:50]):
            rows.append({
                "Chunk ID": chunk["global_chunk_id"],
                "Filename": chunk["filename"],
                "Page": format_page(chunk["page"]),
                "Embedding dimensions": embeddings[i].shape[0],
                "Embedding preview": np.round(embeddings[i][:12], 5).tolist(),
                "Text preview": chunk["text"][:140],
            })

        st.dataframe(rows, use_container_width=True)
        if len(chunks) > 50:
            st.caption("Showing the first 50 embeddings. All embeddings remain available in session state.")


# -----------------------------
# Chat / RAG
# -----------------------------
st.divider()
st.subheader("💬 Ask your documents")

if not st.session_state.chunks:
    st.info("Upload documents and build the RAG index before asking questions.")
else:
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

            if message["role"] == "assistant" and message.get("sources"):
                st.markdown("**Retrieved Sources**")
                for source in message["sources"]:
                    with st.expander(
                        f"📌 {source['filename']} — Page {format_page(source['page'])} "
                        f"— Hybrid {source['hybrid_score']:.3f}"
                    ):
                        st.caption(
                            f"Semantic: {source['semantic_score']:.3f} | "
                            f"Keyword: {source['keyword_score']:.3f}"
                        )
                        st.write(source["text"])

    question = st.chat_input("Ask a question about your documents...")

    if question:
        st.session_state.chat_history.append({
            "role": "user",
            "content": question,
        })

        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching documents and generating answer..."):
                model = load_embedding_model(embedding_model_name)

                retrieved = hybrid_search(
                    question,
                    st.session_state.chunks,
                    st.session_state.embeddings,
                    st.session_state.faiss_index,
                    model,
                    top_k=top_k,
                    semantic_weight=semantic_weight,
                    keyword_weight=keyword_weight,
                )

                if not retrieved:
                    answer = "The requested information is not available in the provided documents."
                else:
                    try:
                        answer = answer_with_groq(
                            question,
                            retrieved,
                            groq_model,
                        )
                    except Exception as exc:
                        answer = f"Unable to generate the answer: {exc}"

                st.markdown(answer)

                st.markdown("**Retrieved Sources**")
                for source in retrieved:
                    with st.expander(
                        f"📌 {source['filename']} — Page {format_page(source['page'])} "
                        f"— Hybrid {source['hybrid_score']:.3f}"
                    ):
                        st.caption(
                            f"Semantic: {source['semantic_score']:.3f} | "
                            f"Keyword: {source['keyword_score']:.3f}"
                        )
                        st.write(source["text"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": answer,
            "sources": retrieved,
        })
