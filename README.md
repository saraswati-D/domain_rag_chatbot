# Domain-Specific RAG Chatbot for PDF Question Answering

A Streamlit chatbot that answers questions from your own uploaded PDFs
(course notes, company policies, manuals, legal documents, training
material) using Retrieval-Augmented Generation (RAG). It only answers
from the retrieved passages and shows the source document + page for
every answer.

## How it works

```
Upload PDF files
      |
Extract text from each page  (pypdf)
      |
Split text into chunks        (LangChain text splitters)
      |
Convert chunks into embeddings (Sentence Transformers: all-MiniLM-L6-v2)
      |
Store embeddings in FAISS
      |
User asks a question
      |
Retrieve top-k similar chunks
      |
Send context + question to the LLM (Groq)
      |
Display answer with source document and page
```

## Project structure

```
domain_rag_chatbot/
├── app.py               # Streamlit UI
├── rag_pipeline.py      # Retrieval + LLM answer generation
├── document_loader.py   # PDF upload validation + text extraction
├── vector_store.py      # Chunking, embeddings, FAISS index
├── prompt.py             # Guardrail prompt template
├── requirements.txt
├── .env.example
├── documents/            # Put sample PDFs here
├── vector_store/saved_index/   # Saved FAISS index (auto-created)
└── tests/test_questions.csv    # Evaluation sheet
```

## Setup

1. **Clone and enter the project**
   ```bash
   git clone <your-repo-url>
   cd domain_rag_chatbot
   ```

2. **Create a virtual environment and install dependencies**
   ```bash
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. **Set up your API key**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and add a free Groq API key from
   https://console.groq.com/keys

4. **Run the app**
   ```bash
   streamlit run app.py
   ```

## Usage

1. Upload one or more PDF files in the sidebar.
2. Click **Process Documents** — this extracts text, chunks it,
   creates embeddings, and builds the FAISS index.
3. Ask questions in the chat box. Each answer shows its source
   document and page number in an expandable "Sources" section.
4. Use **Clear Chat** to reset the conversation without rebuilding
   the index.

If the answer isn't in your documents, the chatbot will say:
> "I could not find this information in the uploaded documents."

## Swapping the LLM provider

By default this project uses **Groq** (`rag_pipeline.py::call_llm`).
To use Gemini, OpenAI, or a local model instead, replace the body of
`call_llm` with the equivalent API call — the rest of the pipeline
(retrieval, prompting, source display) stays the same.

## Testing

`tests/test_questions.csv` contains a starter set of 15 questions to
evaluate the chatbot against your own documents. For each question,
fill in:
- **Retrieved Source** — what the system actually retrieved
- **Correct?** — yes/no, based on whether the answer was accurate and grounded

Evaluate on:
- Retrieval accuracy (right document/page found?)
- Answer correctness (supported by retrieved context?)
- Groundedness (no unsupported claims?)
- Refusal quality (says "not found" when appropriate?)
- Response time

## Responsible AI & security notes

- API keys are kept in `.env` (never committed — see `.gitignore`).
- The prompt instructs the model to ignore any instructions embedded
  inside uploaded documents.
- Generated answers may still be wrong — verify high-stakes
  information independently.
- Only upload documents you have permission to use.
