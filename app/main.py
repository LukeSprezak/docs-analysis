from fastapi import FastAPI

from app.knowledge_management.ui.api.routers import chat, documents, qa, summarize
from app.shared.config import settings
from app.shared.logging import setup_logging

setup_logging()

app = FastAPI(title=settings.PROJECT_NAME, openapi_url=f"{settings.API_V1_STR}/openapi.json")

app.include_router(documents.router, prefix=settings.API_V1_STR)
app.include_router(summarize.router, prefix=settings.API_V1_STR)
app.include_router(chat.router, prefix=settings.API_V1_STR)
app.include_router(qa.router, prefix=settings.API_V1_STR)


@app.get("/")
async def root():
    return {"message": f"Welcome to {settings.PROJECT_NAME} API"}


@app.get("/health")
async def health():
    return {"status": "ok"}
