"""
knowledge_base.py
RAG knowledge base using Ollama embeddings and numpy cosine similarity.
Keeps everything local — no cloud APIs or heavy vector-DB dependencies.
"""
import csv
import logging
import os
from typing import List, Tuple

import numpy as np

from config import KNOWLEDGE_BASE_PATH, OLLAMA_HOST, OLLAMA_EMBED_MODEL

log = logging.getLogger(__name__)


class KnowledgeBase:
    def __init__(self):
        self.documents: List[str] = []
        self.raw_rows: List[dict] = []
        self._embeddings: np.ndarray | None = None
        self._summary: str = ""
        self._embed_available = False

        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_HOST)
            self._client.list()
            self._embed_available = True
        except Exception as exc:
            log.warning("Ollama not reachable — KB semantic search disabled: %s", exc)
            self._client = None

        self._load_data()

    # ── Data loading ──────────────────────────────────────────────────────────

    def _load_data(self):
        if not os.path.isfile(KNOWLEDGE_BASE_PATH):
            log.warning("Knowledge base file not found: %s", KNOWLEDGE_BASE_PATH)
            return

        with open(KNOWLEDGE_BASE_PATH, newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                self.raw_rows.append(row)
                self.documents.append(self._row_to_text(row))

        if not self.documents:
            log.warning("Knowledge base is empty.")
            return

        self._summary = self._compute_summary()
        log.info("Loaded %d documents from knowledge base.", len(self.documents))

        if self._embed_available:
            try:
                self._embeddings = self._embed_batch(self.documents)
                log.info("Embedded %d documents (%s).",
                         len(self.documents), OLLAMA_EMBED_MODEL)
            except Exception as exc:
                log.warning("Embedding failed — falling back to keyword search: %s", exc)
                self._embed_available = False

    @staticmethod
    def _row_to_text(row: dict) -> str:
        return (
            f"Application {row.get('Application ID', 'N/A')} "
            f"named '{row.get('App Name', 'N/A')}' "
            f"is classified as {row.get('Application Tier', 'N/A')} "
            f"with {row.get('Monitoring Level', 'N/A')} monitoring level. "
            f"Its average incident time-to-resolution is "
            f"{row.get('Incident TTR (hrs)', 'N/A')} hours "
            f"and its observability risk score is "
            f"{row.get('Observability Risk Score', 'N/A')}."
        )

    def _compute_summary(self) -> str:
        if not self.raw_rows:
            return "No data available."

        tier_counts: dict[str, int] = {}
        monitoring_counts: dict[str, int] = {}
        risk_scores: list[float] = []
        ttr_values: list[float] = []

        for r in self.raw_rows:
            tier = r.get("Application Tier", "Unknown")
            tier_counts[tier] = tier_counts.get(tier, 0) + 1

            ml = r.get("Monitoring Level", "Unknown")
            monitoring_counts[ml] = monitoring_counts.get(ml, 0) + 1

            try:
                risk_scores.append(float(r["Observability Risk Score"]))
            except (KeyError, ValueError):
                pass
            try:
                ttr_values.append(float(r["Incident TTR (hrs)"]))
            except (KeyError, ValueError):
                pass

        n = len(self.raw_rows)
        tiers = ", ".join(f"{k}: {v}" for k, v in sorted(tier_counts.items()))
        mons = ", ".join(f"{k}: {v}" for k, v in sorted(monitoring_counts.items()))
        avg_risk = np.mean(risk_scores) if risk_scores else 0
        avg_ttr = np.mean(ttr_values) if ttr_values else 0
        max_risk_idx = int(np.argmax(risk_scores)) if risk_scores else 0
        min_risk_idx = int(np.argmin(risk_scores)) if risk_scores else 0

        return (
            f"The knowledge base contains {n} applications.\n"
            f"Tier distribution: {tiers}.\n"
            f"Monitoring levels: {mons}.\n"
            f"Average observability risk score: {avg_risk:.1f} "
            f"(range {min(risk_scores):.0f}–{max(risk_scores):.0f}).\n"
            f"Highest risk: {self.raw_rows[max_risk_idx].get('App Name')} "
            f"(score {max(risk_scores):.0f}).\n"
            f"Lowest risk: {self.raw_rows[min_risk_idx].get('App Name')} "
            f"(score {min(risk_scores):.0f}).\n"
            f"Average incident TTR: {avg_ttr:.1f} hours "
            f"(range {min(ttr_values):.1f}–{max(ttr_values):.1f} hrs)."
        )

    # ── Embeddings ────────────────────────────────────────────────────────────

    def _embed_batch(self, texts: List[str]) -> np.ndarray:
        response = self._client.embed(model=OLLAMA_EMBED_MODEL, input=texts)
        return np.array(response["embeddings"], dtype=np.float32)

    # ── Query ─────────────────────────────────────────────────────────────────

    def query(self, question: str, top_k: int = 5) -> List[str]:
        if not self.documents:
            return []

        if self._embed_available and self._embeddings is not None:
            return self._semantic_search(question, top_k)
        return self._keyword_search(question, top_k)

    def _semantic_search(self, question: str, top_k: int) -> List[str]:
        q_emb = self._embed_batch([question])
        norms_d = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
        norms_q = np.linalg.norm(q_emb, axis=1, keepdims=True)
        cos_sim = (self._embeddings @ q_emb.T) / (norms_d * norms_q + 1e-10)
        cos_sim = cos_sim.flatten()

        k = min(top_k, len(self.documents))
        top_idx = np.argsort(cos_sim)[-k:][::-1]
        return [self.documents[i] for i in top_idx]

    def _keyword_search(self, question: str, top_k: int) -> List[str]:
        """Simple fallback: score documents by overlapping words."""
        q_words = set(question.lower().split())
        scored: List[Tuple[int, int]] = []
        for i, doc in enumerate(self.documents):
            doc_words = set(doc.lower().split())
            scored.append((len(q_words & doc_words), i))
        scored.sort(reverse=True)
        return [self.documents[idx] for _, idx in scored[:top_k]]

    # ── Public helpers ────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        return self._summary

    def doc_count(self) -> int:
        return len(self.documents)
