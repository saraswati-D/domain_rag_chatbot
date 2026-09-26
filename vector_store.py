"""
vector_store.py
----------------
Splits extracted page text into overlapping chunks, embeds them with
a Sentence Transformers model, and stores/searches them in a FAISS
vector index. Metadata (source doc + page number) travels alongside
each chunk so retrieval results can cite where they came from.
"""

import os
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"

INDEX_DIR = "vector_store/saved_index"
INDEX_PATH = os.path.join(INDEX_DIR, "index.faiss")
META_PATH = os.path.join(INDEX_DIR, "metadata.pkl")


class VectorStore:
    def __init__(self):
        self.model = SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.index = None
        self.chunks = []  # list of {"text", "source", "page"}

    # ---------- Chunking ----------
    def chunk_pages(self, pages):
        """
        pages: list of {"text", "source", "page"} from document_loader
        Returns a list of chunk dicts with the same metadata, one
        entry per chunk (a page may produce several chunks).
        """
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

        chunks = []
        for page in pages:
            pieces = splitter.split_text(page["text"])
            for piece in pieces:
                chunks.append({
                    "text": piece,
                    "source": page["source"],
                    "page": page["page"],
                })
        return chunks

    # ---------- Embeddings + Index ----------
    def build_index(self, pages):
        """
        Chunk the given pages, embed every chunk, and build a fresh
        FAISS index (cosine similarity via normalized inner product).
        """
        self.chunks = self.chunk_pages(pages)
        if not self.chunks:
            raise ValueError("No text could be extracted from the uploaded documents.")

        texts = [c["text"] for c in self.chunks]
        embeddings = self.model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
        embeddings = self._normalize(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)  # inner product on normalized vectors == cosine similarity
        self.index.add(embeddings)

    def _normalize(self, vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-10
        return vectors / norms

    # ---------- Retrieval ----------
    def search(self, query, top_k=4):
        """
        Embed the query and return the top_k most similar chunks,
        each with a similarity score and its source metadata.
        """
        if self.index is None or not self.chunks:
            return []

        query_vec = self.model.encode([query], convert_to_numpy=True)
        query_vec = self._normalize(query_vec)

        scores, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            chunk = self.chunks[idx]
            results.append({
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"],
                "score": float(score),
            })
        return results

    # ---------- Persistence ----------
    def save(self):
        os.makedirs(INDEX_DIR, exist_ok=True)
        faiss.write_index(self.index, INDEX_PATH)
        with open(META_PATH, "wb") as f:
            pickle.dump(self.chunks, f)

    def load(self):
        if not (os.path.exists(INDEX_PATH) and os.path.exists(META_PATH)):
            return False
        self.index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "rb") as f:
            self.chunks = pickle.load(f)
        return True
