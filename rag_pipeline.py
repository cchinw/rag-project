import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
load_dotenv()

#Step 1: Load and chunk the doc
loader = TextLoader("document.txt")
documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
  chunk_size= 200,
  chunk_overlap = 20
)

chunks = splitter.split_documents(documents)
print(f"Loaded ad split into {len(chunks)} chunks")

#Step 2: Load existing vector store
embeddings = OpenAIEmbeddings(
  api_key=os.getenv("OPENAI_API_KEY")
)

vectorstore= Chroma(
  persist_directory="./chroma_db",
  embedding_function=embeddings
)
print("Vector store loaded")

#Step 3: Set up Claude as the LLM
llm = ChatAnthropic(
  model="claude-opus-4-6",
  api_key=os.getenv("ANTHROPIC_API_KEY"),
  temperature=0
)
print("Claude Ready")

#Step 4 create the RAG Chain
def ask(question):
  #Retrieval relevant chunks
  results = vectorstore.similarity_search(question, k=5)

  # Build context from chunks
  context = "\n\n".join([r.page_content for r in results])

  #Send to Claude with context
  messages = [
    SystemMessage(content="You are a helpful assistant. Answer questionsusing only the context provided. If the answer is not in the context, say so"),
    HumanMessage(content=f"Context\n{context}\n\nQuestion: {question}")
  ]
  response = llm.invoke(messages)
  return response.content


#Step 5: Ask Questions
questions = [
  "What has Chinwendu built at Kyndryl?",
  "What AI technologies is Chinwendu learning?",
  "What business did Chinwendu found?",
  "Why does Chinwendu want to work in AI safety?"
]

for question in questions:
  print(f"Question: {question}")
  print(f"Answer: {ask(question)}")
  print('---')