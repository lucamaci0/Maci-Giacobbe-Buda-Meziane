import os
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PDFMinerLoader
 
 
 
use_miner_loader = True
 
 
 
 
# Avvia prima Qdrant con Docker:
# docker run -d -p 6333:6333 qdrant/qdrant
 
client = QdrantClient(host="localhost", port=6333)
 
 
CURRENT_FILE_PATH = os.path.abspath(__file__)
CURRENT_DIRECTORY_PATH = os.path.dirname(CURRENT_FILE_PATH)
DOTENV_PATH = "C/Users/LH668YN/OneDrive - EY/Desktop/Local_RAG/.env"
load_dotenv()
 
print('----------------------------------------------')
print(os.environ["AZURE_API_KEY"])
print('----------------------------------------------')
 
faiss_0__or__qdrant_1 = int(os.environ["FAISS_0__OR__QDRANT_1"])
 
azure_openai_key = os.getenv("AZURE_API_KEY") or ""
azure_openai_endpoint = os.getenv("AZURE_API_BASE") or ""
api_version = os.getenv("AZURE_API_VERSION") or ""
 
deployment_embedding = os.getenv("DEPLOYMENT_EMBEDDING") or ""
 
 
# Modello embedding
embeddings = AzureOpenAIEmbeddings(model="text-embedding-ada-002")
 
 
if use_miner_loader:
   loader = PDFMinerLoader(f"{CURRENT_DIRECTORY_PATH}/knowledge_base/EU AI Act.pdf")
else:
    loader = PyPDFLoader(f"{CURRENT_DIRECTORY_PATH}/knowledge_base/EU AI Act.pdf")
 
docs = loader.load()
 
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,      # max tokens/characters per chunk
    chunk_overlap=200,    # overlap between chunks to preserve context
)
 
# Split the PDF docs into smaller chunks
split_docs = text_splitter.split_documents(docs)
 
####################### SALVATAGGIO DEL VECTOR STORE ############################################
 
 
if faiss_0__or__qdrant_1 == 0:
   
    vector_store = FAISS.from_documents(
        documents=split_docs,
        embedding=embeddings
    )
    vector_store.save_local(f"{CURRENT_DIRECTORY_PATH}/indices/faiss_index_risposte-sbagliate")
 
 
elif faiss_0__or__qdrant_1 == 1:
   
    vector_store = Qdrant.from_documents(
        documents=split_docs,
        embedding=embeddings,
        location="http://localhost:6333",
        collection_name="index_EU_AI_Act",
    )
 
print(f"Loaded {len(split_docs)} chunks into the vector_store.")
 
 
 