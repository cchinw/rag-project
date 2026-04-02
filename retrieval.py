import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

#Load the existing vector store already built
embeddings = OpenAIEmbeddings(
  api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore = Chroma(
  persist_directory="./chroma_db",
  embedding_function=embeddings
)

print(f"Loaded vector store with {vectorstore._collection.count()} chunks")
print("---")

#Ask questions and find relevant chunks
questions = [
  "What has Chinwendu built at Kyndryl?",
  "What AI technologies is Chinwendu learning?",
  "What business did Chinwendu found?",
  "Why does Chinwendu want to work in AI safety?"
]

for question in questions:
  print(f"Question: {question}")
  results = vectorstore.similarity_search(question, k=2)
  print("Most relevant chunks:")
  for i, result in enumerate(results):
    print(f" Result {i+1}: {result.page_content}")
    print("---")