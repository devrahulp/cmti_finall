from pydantic import BaseModel
from typing import List


class UploadedFileInfo(BaseModel):
    filename: str
    content_type: str
    size: int


class ChatResponse(BaseModel):
    answer: str
