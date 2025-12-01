import React, { useRef } from "react";
import "./ChatInput.css";

function ChatInput({
  value,
  onChange,
  onSend,
  onFilesSelected,
  isSending,
  canSend
}) {
  const fileInputRef = useRef(null);

  function handleSubmit(e) {
    e.preventDefault();
    if (!canSend || isSending) return;
    onSend();
  }

  function handleFileChange(e) {
    const selected = Array.from(e.target.files || []);
    onFilesSelected(selected);
  }

  return (
    <form className="chat-input-row" onSubmit={handleSubmit}>
      <button
        type="button"
        className="file-upload-btn"
        onClick={() => fileInputRef.current?.click()}
      >
        📎
      </button>
      <input
        ref={fileInputRef}
        type="file"
        multiple
        onChange={handleFileChange}
        accept=".pdf,.txt,.doc,.docx,image/*"
        style={{ display: "none" }}
      />
      <textarea
        className="chat-input"
        rows={1}
        placeholder="Message ChatGPT…"
        value={value}
        onChange={(e) => onChange(e.target.value)}
      />
      <button
        type="submit"
        className="send-btn"
        disabled={!canSend || isSending}
      >
        {isSending ? "…" : "➤"}
      </button>
    </form>
  );
}

export default ChatInput;
