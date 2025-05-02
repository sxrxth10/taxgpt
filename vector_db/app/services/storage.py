from langchain_community.vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from ..core.config import settings
import os


embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001",
                                          google_api_key=settings.GOOGLE_API_KEY)

vector_db_path = "vector_store"

if os.path.exists(vector_db_path):
    vector_store = FAISS.load_local(vector_db_path,
                                    embeddings=embeddings,
                                    allow_dangerous_deserialization=True)
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"score_threshold": 0.5}
    )
else:
    raise ValueError(f"Vector database not found at {vector_db_path}")


retriever = vector_store.as_retriever(
    search_type="similarity_score_threshold",
    search_kwargs={"score_threshold": 0.5}
)
