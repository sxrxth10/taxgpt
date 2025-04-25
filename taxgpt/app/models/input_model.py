from pydantic import BaseModel

class InputModel(BaseModel):
    question: str
