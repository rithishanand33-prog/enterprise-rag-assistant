
# Enterprise RAG AI Assistant

An AI-powered document question-answering application that allows users to upload PDF documents and ask questions about their contents.

The application uses Retrieval-Augmented Generation (RAG) to retrieve relevant information from uploaded documents and generate answers grounded in that information.

## Features

- Upload multiple PDF documents.
- Extract text from PDFs.
- Split documents into smaller text chunks.
- Generate semantic embeddings using Hugging Face.
- Store and search document embeddings using FAISS.
- Generate answers using Google's Gemini model.
- Display source document and page references.
- Interactive chat interface built with Streamlit.
- Automated tests for core RAG functionality.

## Technology Stack

- Python
- LangChain
- Streamlit
- Google Gemini
- Hugging Face Sentence Transformers
- FAISS
- PyPDF
- Pytest

## Architecture

1. User uploads PDF documents.
2. PyPDF extracts document text.
3. RecursiveCharacterTextSplitter divides text into chunks.
4. Hugging Face generates embeddings.
5. FAISS performs similarity search.
6. Relevant document chunks are passed to Gemini.
7. Gemini generates a grounded answer with source references.

## Project Structure

```text
enterprise-rag-assistant/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── data/
│   └── employee_handbook.pdf
│
├── src/
│   ├── ingest.py
│   ├── chunk.py
│   ├── embed.py
│   ├── vectorstore.py
│   ├── retriever.py
│   ├── retrieve.py
│   ├── llm.py
│   └── rag.py
│
└── tests/
    └── test_rag.py
```

## Installation

Clone the repository:

```bash
git clone YOUR_GITHUB_REPOSITORY_URL
cd enterprise-rag-assistant
```

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## API Configuration

Create a `.env` file in the project root:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Get your API key from Google AI Studio.

Never upload your `.env` file or share your API key.

## Run the Application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal.

Upload PDF documents, process them, and ask questions about their contents.

## Run Tests

```bash
python -m pytest -v
```

## Sample Questions

- How many annual leave days do employees receive?
- What is the standard lunch break?
- How should employees report an absence?

## Current Limitations

- PDF documents are currently supported.
- Uploaded document embeddings are held in memory during the application session.
- Answer quality depends on document quality and retrieval relevance.

## Disclaimer

The included employee handbook is fictional sample data created for demonstration and testing purposes.
