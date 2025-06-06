from fastapi import APIRouter
from ..services.storage import retriever
from ..models.query_model import QueryModel

router = APIRouter()


@router.post("/vector")
async def get_data(query: QueryModel):
    documents = retriever.get_relevant_documents(query.question)
    return {"document": documents}
