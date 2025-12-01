import React from "react";
import "./EmptyState.css";
import Button from "./Button.jsx";

function EmptyState({ onNewChat }) {
  return (
    <div className="empty-state">
      <h1>ChatGPT-like UI</h1>
      <p>Start a conversation, upload documents or images, and get answers.</p>
      <Button variant="primary" onClick={onNewChat}>
        Start a new chat
      </Button>
    </div>
  );
}

export default EmptyState;
