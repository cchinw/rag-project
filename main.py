import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

#Load environment variables
load_dotenv()

#Step 1: Load the doc
loader = TextLoader("document.txt")
documents = loader.load()

print(f"loaded {len(documents)} documents(s)")
print(f"Document length: {len(documents[0].page_content)} characters")
print("---")

#Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
  chunk_size=200,
  chunk_overlap=20
)

chunks = splitter.split_documents(documents)

print(f"Split into {len(chunks)} chunks")
print("---")

# Step 3: Prit each chunk so we can see what happened
for i, chunk in enumerate(chunks):
  print(f"Chunk {i+ 1}:")
  print(chunk.page_content)
  print("---")