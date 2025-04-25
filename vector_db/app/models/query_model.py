from pydantic import BaseModel

class QueryModel(BaseModel):
    question: str