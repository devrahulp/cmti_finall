from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware

from dotenv import load_dotenv
from pathlib import Path

# Load backend/.env file safely
load_dotenv(dotenv_path=Path(__file__).parent / ".env")

from .models import UploadedFileInfo, ChatResponse
from .services import generate_answer


app = FastAPI(title="ChatGPT Clone Backend", version="1.0.0")


# CORS: allow your React frontend to talk to backend
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root():
    return {"message": "Backend is running. Use /health or /api/chat."}


@app.get("/health")
async def health_check():
    return {"status": "ok"}


@app.post("/api/chat", response_model=ChatResponse)
async def chat_endpoint(
    prompt: str = Form(...),
    conversationId: Optional[str] = Form(None),
    files: List[UploadFile] = File(default_factory=list),
):
    """
    Handles chat requests with optional file uploads.
    The frontend sends multipart/form-data:
      - prompt (string)
      - conversationId (string)
      - files[] (array of files)
    """

    uploaded_files: List[UploadedFileInfo] = []

    # Process uploaded file metadata
    for file in files:
        contents = await file.read()
        uploaded_files.append(
            UploadedFileInfo(
                filename=file.filename,
                content_type=file.content_type or "application/octet-stream",
                size=len(contents),
            )
        )

    print("📩 /api/chat called:")
    print("Prompt:", prompt)
    print("Conversation ID:", conversationId)
    print("Files:", [f.filename for f in uploaded_files])

    # Generate AI answer
    try:
        answer = generate_answer(
            prompt=prompt,
            files=uploaded_files,
            conversation_id=conversationId,
        )
    except Exception as e:
        import traceback
        print("❌ ERROR in generate_answer:", e)
        traceback.print_exc()
        return ChatResponse(
            answer="Backend failed to generate answer. Check server logs."
        )

    return ChatResponse(answer=answer)
