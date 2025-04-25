from fastapi import FastAPI
from app.routes.response import router as response_router

app = FastAPI(
    title="Tax Assistance API",
    version="1.0",
    description="API for tax-related question answering"
)

# Include routes
app.include_router(response_router)


