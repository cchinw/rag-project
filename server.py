import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

load_dotenv()

#Init FastAPI app
app = FastAPI()

#Allow React frontend to talk to this server
app.add_middleware(
  CORSMiddleware,
  allow_origins=["http://localhost:5173"],
  allow_methods=["*"],
  allow_headers=["*"],
)

#Load vector store and LLM once at startup
embeddings = OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))

vectorstore = Chroma(
  persist_directory="./chroma_db",
  embedding_function=embeddings
)

llm = ChatAnthropic(
  model="claude-opus-4-6",
  api_key=os.getenv("ANTHROPIC_API_KEY"),
  temperature=0
)

#Define question request
class QuestionRequest(BaseModel):
  question: str

#Define the /ask endpoint
@app.post("/ask")
async def ask(request: QuestionRequest):
  # Retrieve relevant chunks
  results = vectorstore.similarity_search(request.question, k=5)

  #Build Context
  context = "\n\n".join([r.page_content for r in results])

  #Send to Claude
  messages = [
    SystemMessage(content="You are a helpful assistant. Answer questions using only the context provided. If the answer is not in the context, say so."),
    HumanMessage(content=f"Context:\n{context}\n\nQuestion: {request.question}")
  ]

  response = llm.invoke(messages)
  return {"answer": response.content}