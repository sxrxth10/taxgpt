from fastapi import FastAPI
from .routes.response  import router
import uvicorn
import os
from dotenv import load_dotenv

load_dotenv()

app = FastAPI(
    title="Vector database api",
    version="1.0",
    description="API for tax-related question answering"
)


# Include routes
app.include_router(router)

