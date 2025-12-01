import React, { createContext, useContext, useMemo } from "react";
import { useLocalStorage } from "../hooks/useLocalStorage.js";
import { STORAGE_KEYS } from "../utils/constants.js";
import { generateTitle } from "../utils/generateTitle.js";

const ChatContext = createContext(null);

export function ChatProvider({ children }) {
  const [conversations, setConversations] = useLocalStorage(
    STORAGE_KEYS.CONVERSATIONS,
    []
  );
  const [activeId, setActiveId] = useLocalStorage(STORAGE_KEYS.ACTIVE_ID, null);

  const activeConversation = useMemo(
    () => conversations.find((c) => c.id === activeId) || null,
    [conversations, activeId]
  );

  function newChat() {
    const conv = {
      id: `conv-${Date.now()}`,
      title: "New chat",
      createdAt: new Date().toISOString(),
      messages: []
    };
    setConversations((prev) => [conv, ...prev]);
    setActiveId(conv.id);
  }

  function deleteConversation(id) {
    setConversations((prev) => prev.filter((c) => c.id !== id));
    if (activeId === id) {
      const remaining = conversations.filter((c) => c.id !== id);
      setActiveId(remaining.length ? remaining[0].id : null);
    }
  }

  function updateConversation(id, updater) {
    setConversations((prev) =>
      prev.map((c) => {
        if (c.id !== id) return c;
        const updated = updater(c);
        if (
          updated.messages &&
          updated.messages.length > 0 &&
          (!updated.title || updated.title === "New chat")
        ) {
          const firstUserMsg = updated.messages.find((m) => m.role === "user");
          if (firstUserMsg && firstUserMsg.content) {
            updated.title = generateTitle(firstUserMsg.content);
          }
        }
        return updated;
      })
    );
  }

  return (
    <ChatContext.Provider
      value={{
        conversations,
        activeId,
        setActiveId,
        activeConversation,
        newChat,
        deleteConversation,
        updateConversation
      }}
    >
      {children}
    </ChatContext.Provider>
  );
}

export function useChatContext() {
  const ctx = useContext(ChatContext);
  if (!ctx) throw new Error("useChatContext must be used within ChatProvider");
  return ctx;
}
