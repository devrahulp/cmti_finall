import { useState } from "react";
import { useChatContext } from "../context/ChatContext.jsx";
import { sendMessageToBackend } from "../api/chatApi.js";

export function useChat() {
  const { activeConversation, updateConversation } = useChatContext();
  const [isSending, setIsSending] = useState(false);

  async function sendMessage(prompt, files) {
    if (!activeConversation) return;
    const trimmed = (prompt || "").trim();
    if (!trimmed && (!files || files.length === 0)) return;
    if (isSending) return;

    setIsSending(true);
    const convId = activeConversation.id;

    // Add user message
    updateConversation(convId, (conv) => {
      const userMessage = {
        id: `msg-${Date.now()}-user`,
        role: "user",
        content: trimmed || "(sent files only)",
        createdAt: new Date().toISOString(),
        files: (files || []).map((f) => ({
          name: f.name,
          size: f.size,
          type: f.type
        }))
      };
      return { ...conv, messages: [...conv.messages, userMessage] };
    });

    try {
      const answer = await sendMessageToBackend(trimmed, files || [], convId);
      updateConversation(convId, (conv) => {
        const assistantMessage = {
          id: `msg-${Date.now()}-assistant`,
          role: "assistant",
          content: answer,
          createdAt: new Date().toISOString()
        };
        return { ...conv, messages: [...conv.messages, assistantMessage] };
      });
    } catch (error) {
      console.error(error);
      updateConversation(convId, (conv) => {
        const errorMessage = {
          id: `msg-${Date.now()}-error`,
          role: "assistant",
          content:
            "There was an error contacting the backend. Please check your API configuration.",
          createdAt: new Date().toISOString()
        };
        return { ...conv, messages: [...conv.messages, errorMessage] };
      });
    } finally {
      setIsSending(false);
    }
  }

  return { activeConversation, sendMessage, isSending };
}
