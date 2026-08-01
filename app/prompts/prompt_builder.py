from app.models.document import Document


class PromptBuilder:

    def build(
        self,
        query: str,
        documents: list[Document],
        history: list = None
    ):

        # ----------------------------
        # Build Context
        # ----------------------------

        context = ""

        for doc in documents:

            context += f"""
Document: {doc.file_name}
Page: {doc.page_number}

{doc.text}

----------------------------------------
"""

        # ----------------------------
        # Conversation History
        # ----------------------------

        history_text = ""

        if history:

            history_text = "Conversation History:\n\n"

            for message in history:

                role = message["role"].capitalize()

                history_text += (
                    f"{role}: {message['content']}\n"
                )

        # ----------------------------
        # Prompt
        # ----------------------------

        prompt = f"""
You are an AI assistant for ABC Technologies.

Answer employee questions ONLY using the provided company documents.

Rules:

1. Use ONLY the retrieved context.

2. Never make up information.

3. Never use outside knowledge.

4. If the answer is unavailable, reply EXACTLY:

"I couldn't find that information in the company's internal documents."

5. Keep answers concise and professional.

6. Do NOT mention document names.

7. Do NOT mention page numbers.

8. Do NOT say "According to the context."

------------------------------------------------------------

{history_text}

Retrieved Context:

{context}

------------------------------------------------------------

Current User Question:

{query}

------------------------------------------------------------

Answer:
"""

        return prompt.strip()