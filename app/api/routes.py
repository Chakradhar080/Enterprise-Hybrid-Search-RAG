import json
import os
import shutil

from fastapi import APIRouter, HTTPException, UploadFile, File
from fastapi.responses import StreamingResponse

from app.models.chat_models import ChatRequest, ChatResponse

router = APIRouter()

# --------------------------------------------------
# Services (Injected from main_api.py)
# --------------------------------------------------

rag_service = None
document_service = None
ingestion_service = None
memory_service = None


# --------------------------------------------------
# Health
# --------------------------------------------------

@router.get("/health")
def health():

    return {
        "status": "healthy"
    }


# --------------------------------------------------
# Chat
# --------------------------------------------------

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):

    global rag_service
    global memory_service

    if rag_service is None:

        raise HTTPException(
            status_code=500,
            detail="RAG Service not initialized."
        )

    memory_service.add_user(request.query)

    response = rag_service.ask(
        query=request.query,
        history=memory_service.get_history()
    )

    memory_service.add_assistant(
        response.answer
    )

    return response


# --------------------------------------------------
# Streaming Chat (Server Sent Events)
# --------------------------------------------------

@router.post("/chat/stream")
def chat_stream(request: ChatRequest):

    global rag_service
    global memory_service

    if rag_service is None:

        raise HTTPException(
            status_code=500,
            detail="RAG Service not initialized."
        )

    memory_service.add_user(request.query)

    def event_generator():

        answer = ""

        for event in rag_service.stream(
            query=request.query,
            history=memory_service.get_history()
        ):

            # Save assistant response

            if event["type"] == "token":

                answer += event["content"]

            # Send SSE event

            yield (
                f"data: {json.dumps(event)}\n\n"
            )

        memory_service.add_assistant(answer)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream"
    )


# --------------------------------------------------
# Upload
# --------------------------------------------------

@router.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...)
):

    global ingestion_service

    upload_dir = "data/raw/Uploaded"

    os.makedirs(
        upload_dir,
        exist_ok=True
    )

    destination = os.path.join(
        upload_dir,
        file.filename
    )

    with open(destination, "wb") as buffer:

        shutil.copyfileobj(
            file.file,
            buffer
        )

    count = ingestion_service.ingest(
        upload_dir
    )

    return {
        "message": "Upload successful",
        "chunks": count
    }


# --------------------------------------------------
# Documents
# --------------------------------------------------

@router.get("/documents")
def get_documents():

    global document_service

    if document_service is None:

        raise HTTPException(
            status_code=500,
            detail="Document Service not initialized."
        )

    return document_service.get_documents()


# --------------------------------------------------
# Clear Memory
# --------------------------------------------------

@router.post("/memory/clear")
def clear_memory():

    global memory_service

    if memory_service is None:

        raise HTTPException(
            status_code=500,
            detail="Memory Service not initialized."
        )

    memory_service.clear()

    return {
        "message": "Conversation cleared."
    }