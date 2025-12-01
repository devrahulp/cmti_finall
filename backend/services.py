from typing import List, Optional
import os
import openai

from .models import UploadedFileInfo


# Load API key from environment (backend/.env)
openai.api_key = os.getenv("OPENAI_API_KEY")


def generate_answer(
    prompt: str,
    files: List[UploadedFileInfo],
    conversation_id: Optional[str] = None,
) -> str:
    """
    Generates an AI answer using OpenAI's ChatCompletion API.
    Includes uploaded file metadata in the prompt.
    """

    # If no valid key, return graceful safe message
    if not openai.api_key:
        file_summary = ""
        if files:
            file_summary = "\n\nUploaded files:\n" + "\n".join(
                [f"- {f.filename} ({f.size} bytes)" for f in files]
            )

        return (
            "❗ OPENAI_API_KEY is not set.\n\n"
            f'Your prompt was:\n"{prompt}"\n'
            f"Conversation ID: {conversation_id or 'none'}"
            f"{file_summary}\n\n"
            "Set OPENAI_API_KEY inside backend/.env to get real AI responses."
        )

    # Add file metadata to the user content
    file_context = ""
    if files:
        file_context = "\n\nUploaded files:\n" + "\n".join(
            [f"- {f.filename} ({f.size} bytes)" for f in files]
        )

    user_content = prompt + file_context

    # Use GPT model (make sure your key has access to this model)
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant. Use file metadata only as context.",
            },
            {"role": "user", "content": user_content},
        ],
    )

    # Extract and return AI’s message
    return response["choices"][0]["message"]["content"]
