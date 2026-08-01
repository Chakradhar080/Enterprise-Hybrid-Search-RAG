from app.llm.ollama_client import OllamaClient


class QueryRewriter:

    def __init__(self):

        self.llm = OllamaClient()

    def rewrite(
        self,
        query: str,
        history=None
    ):

        if not history:
            return query

        conversation = ""

        for msg in history:

            conversation += (
                f"{msg['role'].capitalize()}: "
                f"{msg['content']}\n"
            )

        prompt = f"""
You are a query rewriting assistant.

Your job is to rewrite the CURRENT USER QUESTION into a
fully self-contained search query.

Rules:

- Preserve the user's intent.
- Use the conversation history.
- Do NOT answer the question.
- Return ONLY the rewritten query.

Conversation:

{conversation}

Current Question:

{query}

Rewritten Query:
"""

        rewritten = self.llm.generate(prompt)

        return rewritten.strip()