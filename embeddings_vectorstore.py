import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

#load document
loader = TextLoader("document.txt")
documents = loader.load()
print(f"Loaded {len(documents)} document(s)")

#Split into Chunks
splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  chunk_overlap=20
)

chunks = splitter.split_documents(documents)
print(f"Split into {len(chunks)} chunks")

#Create embeddings and store in vector database
print("Creating embeddings and storing in vector database...")

embeddings = OpenAIEmbeddings(
  api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma.from_documents(
  documents=chunks,
  embedding=embeddings,
  persist_directory="./chroma_db"
)

print(f"Store {vectorstore._collection.count()} chunks in vector database")
print("Vector store ready.")