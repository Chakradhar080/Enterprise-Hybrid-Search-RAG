import re


class TextCleaner:

    @staticmethod
    def clean(text: str) -> str:
        """
        Clean extracted PDF text.
        """

        # Replace multiple spaces
        text = re.sub(r"[ ]+", " ", text)

        # Replace multiple newlines
        text = re.sub(r"\n+", "\n", text)

        # Replace tabs
        text = text.replace("\t", " ")

        # Remove invisible Unicode characters
        text = text.replace("\u200b", "")

        # Remove leading/trailing whitespace
        text = text.strip()

        return text