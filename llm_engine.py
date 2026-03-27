"""
llm_engine.py
Ollama LLM integration for Obi — generates conversational responses
using RAG context from the knowledge base.
"""
import logging
from typing import List

from config import OLLAMA_MODEL, OLLAMA_HOST

log = logging.getLogger(__name__)

_SYSTEM_PROMPT_TEMPLATE = """\
You are Obi, an AI-powered observability assistant for enterprise application \
monitoring.

{kb_summary_block}

STRICT RULES — you MUST follow these without exception:
1. ONLY answer questions using data from the knowledge-base context provided \
below. Do NOT use any outside knowledge, do NOT guess, and do NOT fabricate \
data.
2. If the user's question cannot be answered from the provided context, \
respond ONLY with: "I don't have that information in my knowledge base. \
I can only answer questions about the applications in our observability \
portfolio."
3. Never invent application names, IDs, metrics, or statistics that are not \
present in the context documents.
4. When answering, cite specific values from the context: application names, \
IDs, tiers, Incident TTR, Monitoring Level, and Observability Risk Score.
5. Keep responses concise (2–3 sentences) because they will be spoken aloud \
by a lip-synced avatar.
6. Be proactive: if the data shows a concern (high risk score, high TTR, \
basic monitoring on a critical tier), mention it.
7. Use a professional but friendly tone.\
"""


class LLMEngine:
    def __init__(self, kb_summary: str = ""):
        self._available = False
        self._client = None
        self._system_prompt = self._build_system_prompt(kb_summary)

        try:
            import ollama
            self._client = ollama.Client(host=OLLAMA_HOST)
            models = self._client.list()
            names = [m.model for m in models.models] if hasattr(models, "models") else []
            found = any(OLLAMA_MODEL in n for n in names)
            if found:
                log.info("Ollama model '%s' is available.", OLLAMA_MODEL)
            else:
                log.warning(
                    "Model '%s' not found (available: %s). "
                    "Run: ollama pull %s",
                    OLLAMA_MODEL, names, OLLAMA_MODEL,
                )
            self._available = True
        except Exception as exc:
            log.error("Cannot connect to Ollama at %s: %s", OLLAMA_HOST, exc)

    # ── Public API 
    def is_available(self) -> bool:
        return self._available

    def generate_response(self, user_query: str,
                          context_docs: List[str]) -> str:
        if not self._available:
            return ("I'm unable to reach the Ollama server. "
                    "Please make sure Ollama is running and try again.")

        context_block = "\n".join(f"• {d}" for d in context_docs) \
            if context_docs else "No relevant documents found."

        user_content = (
            f"Relevant knowledge-base context:\n{context_block}\n\n"
            f"User question: {user_query}"
        )

        try:
            resp = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
            )
            return resp["message"]["content"].strip()
        except Exception as exc:
            log.error("Ollama generation error: %s", exc)
            return f"Sorry, I encountered an error generating a response: {exc}"

    def generate_response_stream(self, user_query: str,
                                 context_docs: List[str]):
        """Yield tokens as they arrive from Ollama (streaming mode)."""
        if not self._available:
            yield ("I'm unable to reach the Ollama server. "
                   "Please make sure Ollama is running and try again.")
            return

        context_block = "\n".join(f"• {d}" for d in context_docs) \
            if context_docs else "No relevant documents found."

        user_content = (
            f"Relevant knowledge-base context:\n{context_block}\n\n"
            f"User question: {user_query}"
        )

        try:
            stream = self._client.chat(
                model=OLLAMA_MODEL,
                messages=[
                    {"role": "system", "content": self._system_prompt},
                    {"role": "user", "content": user_content},
                ],
                stream=True,
            )
            for chunk in stream:
                token = chunk.get("message", {}).get("content", "")
                if token:
                    yield token
        except Exception as exc:
            log.error("Ollama streaming error: %s", exc)
            yield f"Sorry, I encountered an error: {exc}"

    # ── Internals 

    @staticmethod
    def _build_system_prompt(kb_summary: str) -> str:
        if kb_summary:
            block = (
                "Knowledge-base summary (always available):\n"
                f"{kb_summary}"
            )
        else:
            block = "No knowledge-base summary is loaded."
        return _SYSTEM_PROMPT_TEMPLATE.format(kb_summary_block=block)
