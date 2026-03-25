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
monitoring. You help users understand their application portfolio's health, \
risk levels, incident response times, and monitoring gaps.

{kb_summary_block}

Guidelines:
- When context documents are provided, use them to give specific, \
data-driven answers. Reference application names, IDs, tiers, risk scores, \
and TTR values.
- Keep responses concise (2–4 sentences) because they will be spoken aloud \
by a lip-synced avatar.
- Be proactive: suggest improvements such as upgrading monitoring for \
high-risk or high-TTR applications.
- If the question cannot be answered from the knowledge base, say so clearly.
- Use a professional but friendly tone.\
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
