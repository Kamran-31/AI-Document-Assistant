# 📚 Advanced RAG Document Assistant

A professional Streamlit-based AI Document Assistant implementing an end-to-end **Advanced Retrieval-Augmented Generation (RAG)** pipeline.

It supports local **PDF, DOCX, TXT, and Markdown** documents, plus directly downloadable public/shared Google Drive files.

## Features

### 1. Multi-format document extraction

Supported formats:

- PDF
- DOCX
- TXT
- MD / Markdown

Each format has a separate extraction function:

- `extract_pdf()`
- `extract_docx()`
- `extract_txt()`
- `extract_md()`

The pipeline preserves:

- Filename
- Page number where available
- Source type
- File hash

PDF extraction is page-aware. DOCX, TXT, and MD generally do not contain reliable page metadata through the extraction libraries used here, so their page value is shown as `N/A`.

### 2. Overlapping text chunking

Extracted text is split into manageable overlapping word-based chunks.

Default:

- Chunk size: 900 words
- Chunk overlap: 150 words

Every chunk keeps:

- Chunk ID
- Filename
- Page metadata
- Source document metadata
- Original chunk text

The chunk size and overlap can be adjusted from the sidebar.

### 3. Sentence Transformers embeddings

Document chunks are embedded using Sentence Transformers.

Default model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Another available option is:

```text
sentence-transformers/all-mpnet-base-v2
```

Embeddings are generated **once when the RAG index is built**, not every time a user asks a question.

The app stores the embedding matrix in Streamlit session state and displays:

- Embedding dimensions
- Filename
- Page
- Chunk ID
- Embedding preview
- Text preview

For each question, only the **question embedding** is generated.

### 4. FAISS semantic retrieval

The app uses FAISS `IndexFlatIP` with normalized embeddings.

Because embeddings are normalized, inner product provides cosine-similarity-style semantic retrieval.

The user's question is embedded and compared with the document chunk embeddings to retrieve semantically relevant chunks.

### 5. Keyword retrieval

A lightweight keyword retrieval layer extracts important terms from the question.

Common stopwords are removed, and chunks receive a keyword-overlap score based on important query terms.

This helps retrieve chunks where exact terminology matters even if semantic similarity is weaker.

### 6. Hybrid retrieval

Semantic and keyword search are combined.

The default ranking is:

```text
Hybrid Score =
    0.70 × Semantic Score
  + 0.30 × Keyword Score
```

The semantic/keyword weighting can be changed in the sidebar.

Retrieved chunks preserve their original metadata.

### 7. Groq-powered answer generation

Groq is used as the generation layer.

The LLM receives:

- User question
- Retrieved chunks
- Filename
- Page metadata

The system prompt instructs the model to:

- Answer only from retrieved context
- Avoid outside knowledge
- Avoid hallucinating
- State when information is unavailable

If the supplied context does not contain the answer, the assistant responds:

> The requested information is not available in the provided documents.

### 8. Source transparency

Every generated answer displays the retrieved sources underneath it.

Each source includes:

- Filename
- Page number when available
- Hybrid score
- Semantic score
- Keyword score
- Retrieved text chunk

This makes the RAG pipeline inspectable.

### 9. Google Drive source

The app provides a Google Drive input where users can paste a public/shared Drive **file link**.

The supported pipeline is:

```text
Google Drive
     ↓
Download file
     ↓
Extract text
     ↓
Chunk
     ↓
Sentence Transformer embedding
     ↓
FAISS + Keyword Search
     ↓
Hybrid Retrieval
     ↓
Groq
```

The same pipeline is used for local uploads and Drive files.

#### Google Drive limitation

This implementation intentionally avoids hardcoded Google credentials and OAuth configuration.

It supports directly downloadable **public/shared file links**.

Arbitrary private folder enumeration is not possible from a folder URL alone without using the Google Drive API/OAuth or a service account. A future production version can add Drive API authentication to enumerate folders recursively.

### 10. Processing optimization

The app avoids unnecessary document reprocessing.

It uses:

- Streamlit session state
- `st.cache_resource` for the embedding model
- `st.cache_resource` for the Groq client
- SHA-256 file hashes
- A processing signature based on:
  - Document hashes
  - Chunk size
  - Chunk overlap
  - Embedding model

Document embeddings are not recreated for every question.

The sequence is:

```text
Documents changed?
      │
      ├── No → Reuse existing chunks + embeddings + FAISS index
      │
      └── Yes → Rebuild index
```

## Installation

Create a virtual environment:

Install dependencies:

```bash
pip install -r requirements.txt
```

## Configure Groq API key

The application does **not** hardcode the Groq API key.

For Streamlit, create:

```text
.streamlit/secrets.toml
```

Add:

```toml
GROQ_API_KEY = "your_groq_api_key_here"
```

Alternatively, the app can read `GROQ_API_KEY` from the environment.

Do not commit `.streamlit/secrets.toml` to Git.

Recommended `.gitignore` entry:

```gitignore
.streamlit/secrets.toml
.venv/
__pycache__/
```

## Run the application

```bash
streamlit run app.py
```

Then open the Streamlit URL shown in the terminal.

## Recommended workflow

1. Upload PDF/DOCX/TXT/MD documents.
2. Optionally add a public/shared Google Drive file.
3. Adjust chunk size and overlap if necessary.
4. Click **Process / Rebuild RAG Index**.
5. Review extracted document information.
6. Review the created chunks.
7. Inspect the generated embeddings.
8. Ask questions in the chat interface.
9. Review the retrieved sources under every answer.

## Architecture

```text
             ┌─────────────────────┐
             │ Local Uploads        │
             │ PDF/DOCX/TXT/MD      │
             └──────────┬──────────┘
                        │
             ┌──────────▼──────────┐
             │ Google Drive Files  │
             └──────────┬──────────┘
                        │
                        ▼
              ┌──────────────────┐
              │ Document         │
              │ Extraction       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Chunking         │
              │ + Metadata       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ Sentence         │
              │ Transformers     │
              │ Embeddings       │
              └────────┬─────────┘
                       │
                       ▼
              ┌──────────────────┐
              │ FAISS Vector     │
              │ Index            │
              └────────┬─────────┘
                       │
User Question ─────────┼──────────────┐
                       │              │
                       ▼              ▼
              ┌──────────────┐ ┌──────────────┐
              │ Semantic     │ │ Keyword      │
              │ Search       │ │ Search       │
              └──────┬───────┘ └──────┬───────┘
                     │                │
                     └───────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │ Hybrid Ranking  │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Retrieved       │
                    │ Context         │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Groq LLM        │
                    │ Context-only    │
                    │ Generation      │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ Answer + Sources│
                    └─────────────────┘
```

## Important implementation notes

### PDF page metadata

PDF pages are extracted individually, so retrieved PDF chunks retain their page number.

### DOCX page metadata

A standard DOCX file does not reliably expose rendered page boundaries through `python-docx`. Therefore, DOCX chunks have `page = N/A`.

### TXT/MD page metadata

Plain-text and Markdown files do not inherently contain page numbers, so their page metadata is also `N/A`.

### Embedding reuse

The application intentionally does not call the embedding model on all document chunks when a user asks a question.

At question time:

```text
Question
   ↓
One query embedding
   ↓
FAISS search against existing document embeddings
```

The document embeddings remain in session state.

### Production considerations

For a production deployment, consider adding:

- Persistent vector storage
- A database for document metadata
- Google Drive API/OAuth
- Recursive folder ingestion
- OCR for scanned PDFs
- Reranking with a cross-encoder
- Hybrid BM25 + dense retrieval
- Metadata filtering
- Document-level access control
- Conversation-aware retrieval
- Query rewriting
- Parent-child retrieval
- Context compression
- Observability/tracing
- Background ingestion jobs
- Persistent embedding cache
- Rate-limit and error handling
- File size/type security validity 
