
import streamlit as st

from src.ingest import load_uploaded_pdf
from src.chunk import create_chunks
from src.vectorstore import create_vectorstore
from src.llm import get_llm

from src.rag import (
    answer_question,
    handle_conversational_message,
)


st.set_page_config(
    page_title="Enterprise RAG Assistant",
    page_icon="🤖",
    layout="wide"
)


# --------------------------------------------------
# Header
# --------------------------------------------------

st.title("🤖 Enterprise RAG Assistant")

st.caption(
    "AI-powered document question answering using "
    "Retrieval-Augmented Generation."
)

st.divider()


# --------------------------------------------------
# Initialize session state
# --------------------------------------------------

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

if "llm" not in st.session_state:
    st.session_state.llm = None

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "processed_files" not in st.session_state:
    st.session_state.processed_files = []


# --------------------------------------------------
# Sidebar
# --------------------------------------------------

with st.sidebar:

    st.header("📁 Document Management")

    uploaded_files = st.file_uploader(
        "Upload company PDF documents",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:

        st.write("**Selected documents:**")

        for file in uploaded_files:
            st.write(f"📄 {file.name}")

    process_button = st.button(
        "⚙️ Process Documents",
        use_container_width=True
    )

    if st.button(
        "🗑️ Clear Chat",
        use_container_width=True
    ):
        st.session_state.chat_history = []
        st.rerun()

    st.divider()

    st.caption("Supported format: PDF")
    st.caption("Powered by Gemini, LangChain and FAISS")


# --------------------------------------------------
# Process documents
# --------------------------------------------------

if process_button:

    if not uploaded_files:

        st.warning(
            "Please upload at least one PDF document."
        )

    else:

        try:

            with st.spinner("Processing documents..."):

                all_documents = []

                for uploaded_file in uploaded_files:

                    documents = load_uploaded_pdf(
                        uploaded_file
                    )

                    all_documents.extend(documents)

                chunks = create_chunks(all_documents)

                vectorstore = create_vectorstore(chunks)

                llm = get_llm()

                st.session_state.vectorstore = vectorstore
                st.session_state.llm = llm

                # Start a fresh conversation for newly processed documents.
                st.session_state.chat_history = []

                st.session_state.processed_files = [
                    file.name
                    for file in uploaded_files
                ]

            st.success(
                f"Successfully processed {len(uploaded_files)} PDF(s) "
                f"and created {len(chunks)} chunks!"
            )

        except Exception as error:

            st.error(
                f"Document processing failed: {error}"
            )


# --------------------------------------------------
# Main chat area
# --------------------------------------------------

st.subheader("💬 Chat with Your Documents")


if st.session_state.vectorstore is None:

    st.info(
        "You can chat with me now. To ask questions about "
        "company documents, upload PDF files in the sidebar "
        "and process them."
    )

else:

    st.success("Your documents are ready!")

    with st.expander("📚 Processed Documents"):

        for filename in st.session_state.processed_files:

            st.write(f"📄 {filename}")


# --------------------------------------------------
# Display previous conversation
# --------------------------------------------------

for message in st.session_state.chat_history:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])


# --------------------------------------------------
# Chat input
# --------------------------------------------------

question = st.chat_input(
    "Ask a question or say hello..."
)


if question:

    # Preserve earlier messages before adding the current question.
    previous_history = list(
        st.session_state.chat_history
    )

    st.session_state.chat_history.append(
        {
            "role": "user",
            "content": question
        }
    )

    with st.chat_message("user"):

        st.markdown(question)

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                if st.session_state.vectorstore is None:

                    # Allow simple conversation before documents are uploaded.
                    answer = handle_conversational_message(
                        question
                    )

                    if answer is None:

                        answer = (
                            "Please upload and process your "
                            "company PDF documents first. "
                            "Then I can answer questions "
                            "about their contents."
                        )

                else:

                    # Supply previous turns so follow-up questions
                    # can be resolved using conversation context.
                    answer = answer_question(
                        question,
                        st.session_state.vectorstore,
                        st.session_state.llm,
                        chat_history=previous_history,
                    )

                st.markdown(answer)

                st.session_state.chat_history.append(
                    {
                        "role": "assistant",
                        "content": answer
                    }
                )

            except Exception as error:

                st.error(
                    f"Could not generate an answer: {error}"
                )
