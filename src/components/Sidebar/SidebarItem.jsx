import React from "react";
import "./SidebarItem.css";

function SidebarItem({ conversation, active, onSelect, onDelete }) {
  return (
    <button
      className={
        "sidebar-item " + (active ? "sidebar-item-active" : "")
      }
      onClick={onSelect}
    >
      <span className="sidebar-item-title">
        {conversation.title || "Untitled chat"}
      </span>
      <span
        className="sidebar-item-delete"
        onClick={(e) => {
          e.stopPropagation();
          onDelete();
        }}
      >
        ✕
      </span>
    </button>
  );
}

export default SidebarItem;
