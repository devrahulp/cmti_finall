import React from "react";
import "./Sidebar.css";
import { useChatContext } from "../../context/ChatContext.jsx";
import { useAuth } from "../../context/AuthContext.jsx";
import SidebarItem from "./SidebarItem.jsx";
import Avatar from "../Common/Avatar.jsx";
import Button from "../Common/Button.jsx";

function Sidebar() {
  const { user } = useAuth();
  const {
    conversations,
    activeId,
    setActiveId,
    newChat,
    deleteConversation
  } = useChatContext();

  return (
    <aside className="sidebar">
      <div className="sidebar-header">
        <Button variant="secondary" className="full-width" onClick={newChat}>
          + New chat
        </Button>
      </div>

      <div className="sidebar-section">
        <p className="sidebar-label">Recent</p>
        <div className="sidebar-list">
          {conversations.length === 0 && (
            <div className="sidebar-empty">No chats yet</div>
          )}
          {conversations.map((conv) => (
            <SidebarItem
              key={conv.id}
              conversation={conv}
              active={conv.id === activeId}
              onSelect={() => setActiveId(conv.id)}
              onDelete={() => deleteConversation(conv.id)}
            />
          ))}
        </div>
      </div>

      <div className="sidebar-footer">
        <div className="user-pill">
          <Avatar label={user?.name} />
          <div className="user-info">
            <span className="user-name">{user?.name || "User"}</span>
            {user?.email && (
              <span className="user-email">{user.email}</span>
            )}
          </div>
        </div>
      </div>
    </aside>
  );
}

export default Sidebar;
