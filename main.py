from fastapi import FastAPI, UploadFile, File
import os
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_community.vectorstores import Qdrant
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter 
from langchain.chains import RetrievalQA
from langchain_openai import OpenAI

from dotenv import load_dotenv

load_dotenv()

qdrant_client = QdrantClient(
    url = os.getenv('QDRANT_URL'), 
    api_key = os.getenv('QDRANT_KEY')
)
os.environ['OPENAI_API_KEY'] = os.getenv('OPENAI_KEY')
os.environ['COLLECTION_NAME'] = 'my_collection'

embeddings = OpenAIEmbeddings()

vectors_config = models.VectorParams(
    size=1536,
    distance=models.Distance.COSINE
)

# Create collection
# qdrant_client.create_collection(
#     collection_name = os.getenv('COLLECTION_NAME'),
#     vectors_config = vectors_config
# )

# Create vector store
vector_store = Qdrant(
    client=qdrant_client,
    collection_name=os.getenv('COLLECTION_NAME'),
    embeddings=embeddings
)

# Add Documents to the vector store
def get_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator='\n',
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def get_answer(query: str):
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type='stuff',
        retriever=vector_store.as_retriever()
    )

    response = qa.run(query)
    return response

app = FastAPI()  

@app.get("/", description="This is the first route")
async def read():
    return { "message": "server successfully started" }

@app.post("/upload/file/vectorize")
async def vectorize_file(file: UploadFile = File(..., description="A file read as upload file")):
    if not file:
        return {
            "message": "No file found"
        }
    content = await file.read()
    content_str = content.decode('utf-8')
    chunks = get_chunks(content_str)
    vectors = vector_store.add_texts(chunks)
    return {
        "message": "Successfully chunked and vectorised",
        "chunks": chunks,
        "vectors": vectors
    }

@app.get("/query", description="Q&A with your document")
async def qa(query: str):
    response = get_answer(query)
    return { "response": response }