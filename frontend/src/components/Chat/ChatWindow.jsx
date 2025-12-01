import React, { useState } from "react";
import "./ChatWindow.css";
import { useChat } from "../../hooks/useChat.js";
import { useChatContext } from "../../context/ChatContext.jsx";
import MessageBubble from "./MessageBubble.jsx";
import ChatInput from "./ChatInput.jsx";
import EmptyState from "../Common/EmptyState.jsx";

function ChatWindow() {
  const { activeConversation, sendMessage, isSending } = useChat();
  const { newChat } = useChatContext();
  const [input, setInput] = useState("");
  const [files, setFiles] = useState([]);

  if (!activeConversation) {
    return <EmptyState onNewChat={newChat} />;
  }

  function handleFilesSelected(selected) {
    setFiles(selected);
  }

  async function handleSend() {
    const sendFiles = files;
    setFiles([]);
    await sendMessage(input, sendFiles);
    setInput("");
  }

  return (
    <div className="chat-window">
      <header className="chat-header">
        <div>
          <h2>{activeConversation.title || "New chat"}</h2>
          <p>Upload documents, JPEGs and ask questions just like ChatGPT.</p>
        </div>
      </header>

      <div className="chat-messages">
        {activeConversation.messages.length === 0 && (
          <div className="chat-placeholder">
            <p>Ask a question or describe a task to get started.</p>
          </div>
        )}

        {activeConversation.messages.map((m) => (
          <MessageBubble key={m.id} message={m} />
        ))}
      </div>

      <footer className="chat-footer">
        {files.length > 0 && (
          <div className="file-preview-bar">
            {files.map((f, idx) => (
              <div key={idx} className="file-pill">
                <span className="file-name">{f.name}</span>
                <span className="file-size">
                  {(f.size / 1024).toFixed(1)} KB
                </span>
              </div>
            ))}
          </div>
        )}

        <ChatInput
          value={input}
          onChange={setInput}
          onSend={handleSend}
          onFilesSelected={handleFilesSelected}
          isSending={isSending}
          canSend={!!input.trim() || files.length > 0}
        />
        <p className="chat-disclaimer">
          This is a frontend clone. Connect <code>sendMessageToBackend()</code> to
          your own API in <code>src/api/chatApi.js</code>.
        </p>
      </footer>
    </div>
  );
}

export default ChatWindow;
