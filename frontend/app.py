import streamlit as st

from api import (
    stream_chat,
    get_documents,
    upload_pdf,
    clear_memory
)

from components import (
    render_documents,
    render_sources
)

# --------------------------------------------------
# Page Configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Enterprise Hybrid Search RAG",
    page_icon="🤖",
    layout="wide"
)

# --------------------------------------------------
# Session State
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.title("🤖 Hybrid Search RAG")

    try:

        documents = get_documents()

        render_documents(documents)

    except Exception as e:

        st.error(f"Unable to load documents:\n{e}")

    st.divider()

    st.subheader("📤 Upload PDF")

    uploaded_pdf = st.file_uploader(
        "Choose a PDF",
        type=["pdf"]
    )

    if st.button("Upload"):

        if uploaded_pdf is not None:

            with st.spinner("Uploading and indexing..."):

                result = upload_pdf(uploaded_pdf)

            st.success(
                f"Indexed {result['chunks']} chunk(s)."
            )

            st.rerun()

    st.divider()

    if st.button("🗑 Clear Conversation"):

        clear_memory()

        st.session_state.messages = []

        st.rerun()

# --------------------------------------------------
# Main Page
# --------------------------------------------------

st.title("🤖 Enterprise Hybrid Search RAG")

st.caption(
    "Hybrid Search • FAISS • BM25 • Cross Encoder • Ollama"
)

# --------------------------------------------------
# Display Previous Messages
# --------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if (
            message["role"] == "assistant"
            and message.get("sources")
        ):

            render_sources(message["sources"])

# --------------------------------------------------
# User Input
# --------------------------------------------------

question = st.chat_input(
    "Ask about your company documents..."
)

if question:

    # -----------------------------
    # User Message
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    # -----------------------------
    # Assistant Message
    # -----------------------------

    with st.chat_message("assistant"):

        placeholder = st.empty()

        answer = ""

        sources = []

        try:

            for event in stream_chat(question):

                event_type = event.get("type")

                if event_type == "token":

                    answer += event["content"]

                    placeholder.markdown(
                        answer + "▌"
                    )

                elif event_type == "sources":

                    sources = event["content"]

                elif event_type == "done":

                    break

            placeholder.markdown(answer)

            if sources:

                render_sources(sources)

        except Exception as e:

            st.error(str(e))

    # -----------------------------
    # Save Conversation
    # -----------------------------

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources
        }
    )