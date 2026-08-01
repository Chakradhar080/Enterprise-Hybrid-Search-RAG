import streamlit as st


def render_sources(sources):

    st.markdown("### 📚 Sources")

    for source in sources:

        st.info(
            f"📄 {source['file']} (Page {source['page']})"
        )


def render_documents(documents):

    st.subheader("📂 Indexed Documents")

    for doc in documents:

        st.success(
            f"{doc['folder']} / {doc['file']}"
        )