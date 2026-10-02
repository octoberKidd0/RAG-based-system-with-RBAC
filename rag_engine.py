# rag_engine.py

from langchain_ollama import OllamaLLM
from auth import get_allowed_levels   # single source of truth for role permissions

# ── Model setup ───────────────────────────────────────────────────────────────
# num_ctx=1024 instead of 2048 → uses less memory → responds faster.
# We don't need a large context window here because we only pass 3 short chunks.
ANSWER_MODEL = OllamaLLM(
    model="llama3.1",
    temperature=0.1,
    num_ctx=1024,       # reduced from 2048 — main speed improvement
    repeat_penalty=1.1, # slightly lower — reduces hallucinated repetition loops
)


# ── Search query builder ──────────────────────────────────────────────────────
# Problem we're solving:
#   "Tell me about X" → FAISS similarity search for "Tell me about X" often
#   misses documents because the phrasing matches conversational text, not
#   the dense informational text inside the documents.
#
# Solution:
#   Extract the CORE TOPIC from the question (strip filler phrases),
#   then use that as the search query.
#
# We do this with a simple rule-based approach — NO extra LLM call needed.
# That's the main speed win compared to the old build_standalone_question.

import re

# Filler phrases at the start of questions that confuse vector search
_FILLER = re.compile(
    r'^(tell me (about|more about|everything about|something about)|'
    r'what (can you tell me about|do you know about|is|are|does)|'
    r'explain (to me )?|describe |give me (an? )?(overview|summary|explanation) of |'
    r'i want to (know|understand) (about |more about )?|'
    r'could you (explain|describe|tell me about) ?)',
    re.IGNORECASE
)

def extract_search_query(question: str) -> str:
    """
    Strips conversational filler so the vector search hits document content.

    Examples:
      "Tell me about RBAC"          → "RBAC"
      "What is role-based access?"  → "role-based access"
      "How does JWT work?"          → "JWT work"   (already specific)
    """
    cleaned = _FILLER.sub('', question.strip()).strip(' ?.,!')
    # If we stripped too much (< 3 chars) keep the original
    return cleaned if len(cleaned) >= 3 else question


def build_search_query(history: list[dict], question: str) -> str:
    """
    Builds the best possible FAISS search query.

    For the VERY FIRST question (no history): just extract the topic.
    For FOLLOW-UP questions: prepend the last assistant reply's key topic
    so the search is grounded in what was already discussed.

    This replaces the old approach of making a second LLM call just to
    rephrase — that added 5-15 seconds of latency with minimal benefit.
    """
    topic = extract_search_query(question)

    # If there's no history, no special handling needed
    if not history:
        return topic

    # Check if this looks like a follow-up ("more", "that", "it", "them"...)
    follow_up_signals = re.search(
        r'\b(more|that|this|it|them|they|those|the (above|previous|last)|'
        r'continue|elaborate|expand|example|give me|can you)\b',
        question, re.IGNORECASE
    )

    if follow_up_signals:
        # Find the last assistant message to ground the search
        last_assistant = next(
            (m['content'] for m in reversed(history) if m['role'] == 'assistant'),
            None
        )
        if last_assistant:
            # Take just the first sentence of the last reply as context anchor
            first_sentence = last_assistant.split('.')[0][:120]
            return f"{first_sentence}. {topic}"

    return topic


# ── Main query function ───────────────────────────────────────────────────────

def query_with_rbac(
    user_role: str,
    question: str,
    vectorstore,
    history: list[dict] = None
):
    """
    Main entry point called from main.py.

    Returns a dict with two keys:
      "answer"  — the text answer from the LLM
      "sources" — list of {"source": filename, "page": page_or_sheet}

    Returns None if no relevant accessible documents were found.
    """
    if history is None:
        history = []

    allowed_levels = get_allowed_levels(user_role)

    # Step 1: Build a clean search query (NO extra LLM call — this is fast)
    search_query = build_search_query(history, question)
    print(f"[Search query]: {search_query!r}")

    # Step 2: Vector search — fetch more candidates (k=12) so after RBAC
    # filtering we still have enough relevant chunks
    raw_docs = vectorstore.similarity_search(search_query, k=12)

    # Step 3: RBAC filter — discard chunks the user's role cannot see
    filtered = [
        doc for doc in raw_docs
        if doc.metadata.get("access_level") in allowed_levels
    ]

    print(
        f"[RBAC] {len(raw_docs)} retrieved → "
        f"{len(filtered)} accessible to '{user_role}'"
    )

    if not filtered:
        return None

    # Step 4: Use top 4 chunks as context (up from 3 — helps broad questions)
    context_docs = filtered[:4]
    context = "\n\n---\n\n".join(doc.page_content for doc in context_docs)

    # Step 5: Collect source citations (deduplicated by file+page)
    seen_sources = set()
    sources = []
    for doc in context_docs:
        source = doc.metadata.get("source", "Unknown file")
        page   = doc.metadata.get("page", "—")
        key    = (source, str(page))
        if key not in seen_sources:
            seen_sources.add(key)
            sources.append({"source": source, "page": page})

    # Step 6: Build conversation history section for the prompt.
    # This is what enables follow-up questions about the bot's OWN responses —
    # the model can see what it previously said and refer back to it.
    history_section = ""
    if history:
        # Use last 4 exchanges (8 messages) for context — enough for follow-ups
        recent = history[-8:]
        lines  = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
            for m in recent
        )
        history_section = f"PREVIOUS CONVERSATION:\n{lines}\n\n"

    # Step 7: Generate answer
    # Key change: we tell the model it CAN use the conversation history
    # to answer follow-up questions, not just the document excerpts.
    prompt = (
        "You are a helpful document assistant.\n"
        "Answer the user's question using the document excerpts below.\n"
        "You may also refer to the previous conversation if it helps answer a follow-up.\n"
        "If the answer is a list, use bullet points or numbers.\n"
        "Be concise — 3 to 6 sentences, or a clean list.\n"
        "Do not invent facts not found in the excerpts or conversation.\n"
        "If neither the excerpts nor the conversation contain the answer, "
        "reply with exactly: NO_ANSWER\n\n"
        f"{history_section}"
        f"DOCUMENT EXCERPTS:\n{context}\n\n"
        f"QUESTION: {question}\n\n"
        "ANSWER:"
    )

    answer = ANSWER_MODEL.invoke(prompt).strip()

    if "NO_ANSWER" in answer:
        return None

    return {"answer": answer, "sources": sources}