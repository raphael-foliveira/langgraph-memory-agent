from fastapi import APIRouter, Depends
from pydantic import BaseModel
from dependencies.llm import get_chat_service
from services.chat import ChatService

llm_router = APIRouter()


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    response: str | list | dict


SYSTEM_PROMPT = """
You are eager to be an expert about the user. You try to learn as much as you can about them.
"""


@llm_router.post("/chat/{thread_id}", response_model=ChatResponse)
async def chat(
    chat_request: ChatRequest,
    chat_service: ChatService = Depends(get_chat_service),
):
    response = await chat_service.run(chat_request.message)
    return ChatResponse(response=response)
