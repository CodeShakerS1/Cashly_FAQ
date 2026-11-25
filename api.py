"""Cashly chatbot API.

    uvicorn api:app --host 0.0.0.0 --port 8000 --reload
"""
from typing import Literal

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from rag import answer_question

app = FastAPI(title="Cashly Chatbot API")

# Open CORS so the Expo web version can call the API during development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=2000)


class ChatRequest(BaseModel):
    message: str = Field(max_length=500)
    history: list[Message] = Field(default_factory=list, max_length=20)


class ChatResponse(BaseModel):
    answer: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    question = request.message.strip()
    if not question:
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    history = [m.model_dump() for m in request.history]

    try:
        answer = answer_question(question, history)
    except Exception:
        raise HTTPException(status_code=502, detail="The assistant is unavailable right now.")

    return ChatResponse(answer=answer)