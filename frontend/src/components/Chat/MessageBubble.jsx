import React from "react";
import "./MessageBubble.css";
import Avatar from "../Common/Avatar.jsx";
import { formatTime } from "../../utils/formatTime.js";

function MessageBubble({ message }) {
  const isUser = message.role === "user";

  return (
    <div className={"message-row " + (isUser ? "message-row-user" : "")}>
      <Avatar label={isUser ? "You" : "AI"} small />
      <div className="message-bubble">
        <div className="message-meta">
          <span className="message-role">{isUser ? "You" : "Assistant"}</span>
          {message.createdAt && (
            <span className="message-time">{formatTime(message.createdAt)}</span>
          )}
        </div>
        <div className="message-content">
          {message.content}
          {message.files && message.files.length > 0 && (
            <div className="attached-files">
              {message.files.map((f, idx) => (
                <div key={idx} className="attached-file-pill">
                  📎 {f.name}
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;
