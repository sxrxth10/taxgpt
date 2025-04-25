from pydantic import BaseModel

class OutputModel(BaseModel):
    generation: str
