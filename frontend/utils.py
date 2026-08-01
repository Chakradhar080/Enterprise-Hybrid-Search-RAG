import requests

API_URL = "http://127.0.0.1:8000/chat"


def ask_question(question):

    response = requests.post(
        API_URL,
        json={
            "query": question
        }
    )

    response.raise_for_status()

    return response.json()