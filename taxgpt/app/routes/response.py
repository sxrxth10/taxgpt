from fastapi import APIRouter
from app.models.input_model import InputModel
from app.models.output_model import OutputModel
from app.services.workflow import tax_app
from pprint import pprint

router = APIRouter()

@router.post("/response", response_model=OutputModel)
async def get_response(inputs: InputModel):
    initial_state = {
        "question": inputs.question,
        "generation": "",
        "web_search": "No",
        "documents": [],
        "query_type": "",
    }
    for output in tax_app.stream(initial_state):
        for key, value in output.items():
            pprint(f"Node '{key}':")

    return {"generation": value["generation"]}

