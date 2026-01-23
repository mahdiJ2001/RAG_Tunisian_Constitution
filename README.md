# Tunisian Constitution RAG Project

This project is a small experiment I created to **learn about RAG (Retrieval-Augmented Generation)**.  

It uses:

- **FastAPI** for the API server  
- **LangChain + Chroma** for document embeddings and retrieval  
- **AWS Bedrock (Claude 3 Haiku)** for generating answers  
- A PDF of the **Tunisian Constitution** as the knowledge source  

You can ask questions about the constitution, and the system retrieves relevant excerpts to answer your queries.

---

## Usage

1. Place your PDF in `data/uploads/`  
2. Start the FastAPI server:  

```bash
uvicorn main:app --reload
