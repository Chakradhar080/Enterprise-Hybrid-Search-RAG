import json
import requests

from config import API_URL


# --------------------------------------------------
# Streaming Chat
# --------------------------------------------------

def stream_chat(query):

    response = requests.post(
        f"{API_URL}/chat/stream",
        json={
            "query": query
        },
        stream=True
    )

    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):

        if not line:
            continue

        if line.startswith("data: "):

            data = line[6:]          # Remove "data: "

            yield json.loads(data)


# --------------------------------------------------
# Documents
# --------------------------------------------------

def get_documents():

    response = requests.get(
        f"{API_URL}/documents"
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# Upload PDF
# --------------------------------------------------

def upload_pdf(file):

    files = {
        "file": (
            file.name,
            file,
            "application/pdf"
        )
    }

    response = requests.post(
        f"{API_URL}/upload",
        files=files
    )

    response.raise_for_status()

    return response.json()


# --------------------------------------------------
# Clear Memory
# --------------------------------------------------

def clear_memory():

    response = requests.post(
        f"{API_URL}/memory/clear"
    )

    response.raise_for_status()

    return response.json()