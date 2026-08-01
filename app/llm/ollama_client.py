from ollama import chat


class OllamaClient:

    def __init__(self):

        self.model = "llama3.2:3b"

    # --------------------------------------------------
    # Normal Generation
    # --------------------------------------------------

    def generate(
        self,
        prompt: str
    ) -> str:

        response = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )

        return response["message"]["content"]

    # --------------------------------------------------
    # Streaming Generation
    # --------------------------------------------------

    def stream(
        self,
        prompt: str
    ):

        stream = chat(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            stream=True
        )

        for chunk in stream:

            token = chunk["message"]["content"]

            if token:

                yield token