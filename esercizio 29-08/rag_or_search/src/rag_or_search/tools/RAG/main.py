import os
from dotenv import load_dotenv
 
from qdrant_client import QdrantClient
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_community.vectorstores import Qdrant
 
# ---------------------------------------------------------------------
# 1. Load environment variables
# ---------------------------------------------------------------------
load_dotenv()
 
azure_openai_key = os.getenv("AZURE_API_KEY", "None found")
azure_openai_endpoint = os.getenv("AZURE_API_BASE", "None found")
azure_api_version = os.getenv("AZURE_API_VERSION", "None found")
deployment_name = os.getenv("DEPLOYMENT_COMPLETION", "None found")  # chat model deployment
deployment_embedding = os.getenv("DEPLOYMENT_EMBEDDING", "None found")
 
# ---------------------------------------------------------------------
# 2. Initialize embeddings + Qdrant connection
# ---------------------------------------------------------------------
embeddings = AzureOpenAIEmbeddings(
    model=deployment_embedding,
    azure_deployment=deployment_embedding,
    api_version=azure_api_version
)
 
qdrant_client = QdrantClient(host="localhost", port=6333)
 
# Reconnect to the same collection you stored chunks into earlier
vector_store = Qdrant(
    client=qdrant_client,
    collection_name="index_EU_AI_Act",
    embeddings=embeddings,
)
 
# ---------------------------------------------------------------------
# 3. Initialize Chat LLM
# ---------------------------------------------------------------------
llm = AzureChatOpenAI(
    azure_deployment=deployment_name,
    api_version=azure_api_version,
    temperature=0.0  # deterministic answers
)
 
# ---------------------------------------------------------------------
# 4. Interactive QA Loop
# ---------------------------------------------------------------------
def answer_query(user_query: str, k: int = 5):
    """Retrieve context from Qdrant and query the LLM."""
    # Retrieve top-k chunks
    docs = vector_store.similarity_search(user_query, k=k)
    context = "\n\n".join([doc.page_content for doc in docs])
 
    # Build augmented prompt
    system_prompt = (
        "You are a helpful assistant. Use the provided CONTEXT to answer the QUESTION. "
        "If the answer is not in the context, say so explicitly.\n\n"
        f"CONTEXT:\n{context}\n\n"
        f"QUESTION: {user_query}"
    )
 
    response = llm.invoke(system_prompt)
    return {"response": response.content, "context": context}
 
if __name__ == "__main__":
    print("🤖 RAG Q&A with Azure OpenAI + Qdrant. Type 'exit' to quit.\n")
    while True:
        question = input("Your question: ")
        if question.lower() in {"exit", "quit"}:
            break
        answer = answer_query(question)
        print(f"\nAnswer:\n{answer['response']}\n{'-'*80}\n")
        print(f"Context:\n{answer['context']}\n{'-'*80}\n{'#'*80}\n")
 
 