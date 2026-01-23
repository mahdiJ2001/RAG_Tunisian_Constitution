import os
import boto3
from typing import List
from botocore.config import Config
from fastapi import FastAPI
from pydantic import BaseModel
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_aws import ChatBedrock, BedrockEmbeddings

# --- 1. FastAPI SETUP ---
app = FastAPI()

class QuestionRequest(BaseModel):
    question: str

# --- 2. AWS & RAG CONFIGURATION ---
aws_config = Config(retries={'max_attempts': 3, 'mode': 'adaptive'})
bedrock_runtime = boto3.client('bedrock-runtime', config=aws_config)

embeddings = BedrockEmbeddings(model_id="amazon.titan-embed-text-v1", client=bedrock_runtime)

llm = ChatBedrock(
    model="anthropic.claude-3-haiku-20240307-v1:0", 
    client=bedrock_runtime, 
    model_kwargs={"temperature": 0, "max_tokens": 2048}
)

# --- 3. DOCUMENT PROCESSING (Corrected) ---
PDF_PATH = "data/uploads/tunisian_constitution.pdf"
CHROMA_PATH = "./chroma_db"

loader = PyPDFLoader(PDF_PATH)
docs = loader.load()
print(f"✅ PDF Loaded: {len(docs)} pages.")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
doc_splits = text_splitter.split_documents(docs)
print(f"✅ Created {len(doc_splits)} chunks.")

# Initialize Chroma
vectorstore = Chroma(
    collection_name="tn_const", 
    embedding_function=embeddings, 
    persist_directory=CHROMA_PATH
)

# CRITICAL FIX: Check if the database is actually empty
current_count = len(vectorstore.get()['ids'])
print(f"DEBUG: Current database record count: {current_count}")

if current_count == 0:
    print("⚠️ Database is empty! Ingesting chunks now...")
    # Add documents in batches
    batch_size = 20
    for i in range(0, len(doc_splits), batch_size):
        batch = doc_splits[i : i + batch_size]
        vectorstore.add_documents(batch)
        print(f"  > Progress: Added {i + len(batch)}/{len(doc_splits)} chunks")
    print("✅ Ingestion complete.")
else:
    print(f"✅ Using existing database with {current_count} records.")

retriever = vectorstore.as_retriever(search_kwargs={"k": 7})

# --- 4. THE API ENDPOINT ---
@app.post("/ask")
async def ask_constitution(request: QuestionRequest):
    print(f"\n--- [3] RECEIVED QUESTION: {request.question} ---")
    
    # 1. Retrieve the top 'k' documents
    retrieved_docs = retriever.invoke(request.question)
    print(f"DEBUG: Number of relevant chunks found: {len(retrieved_docs)}")
    
    # Check if we actually found anything
    if not retrieved_docs:
        print("DEBUG: ❌ NO DOCUMENTS RETRIEVED!")
        return {
            "question": request.question,
            "answer": "I'm sorry, I couldn't find any relevant sections in the constitution.",
            "references": []
        }

    # 2. Build the context and the reference list
    context_blocks = []
    sources = []
    
    for i, doc in enumerate(retrieved_docs):
        page_label = doc.metadata.get("page", "Unknown")
        human_page = page_label + 1 if isinstance(page_label, int) else "Unknown"
        
        print(f"DEBUG: Using chunk from Page {human_page}")
        
        context_blocks.append(f"--- SOURCE {i+1} (Page {human_page}) ---\n{doc.page_content}")
        
        sources.append({
            "citation": f"[Source {i+1}]",
            "page": human_page,
            "text_snippet": doc.page_content
        })

    full_context = "\n\n".join(context_blocks)

    # 3. Prompt Engineering
    prompt = (
        "You are an expert legal assistant for the Tunisian Constitution. "
        "Use ONLY the provided context to answer. cite using [Source X].\n\n"
        f"CONTEXT:\n{full_context}\n\n"
        f"QUESTION: {request.question}\n"
        "ANSWER:"
    )

    # 4. Invoke LLM
    print("DEBUG: Sending request to AWS Bedrock (Claude 3 Haiku)...")
    response = llm.invoke(prompt)
    print("DEBUG: Response received successfully.")

    return {
        "question": request.question,
        "answer": response.content,
        "references": sources  
    }

@app.get("/")
def read_root():
    return {"status": "Tunisian Constitution RAG API is running"}