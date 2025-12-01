import React from "react";
import { AuthProvider, useAuth } from "../context/AuthContext.jsx";
import { ChatProvider } from "../context/ChatContext.jsx";
import LoginPage from "../components/Auth/LoginPage.jsx";
import Sidebar from "../components/Sidebar/Sidebar.jsx";
import ChatWindow from "../components/Chat/ChatWindow.jsx";

function AppShell() {
  const { user } = useAuth();

  if (!user) {
    return <LoginPage />;
  }

  return (
    <div className="app-shell">
      <ChatProvider>
        <Sidebar />
        <main className="chat-main">
          <ChatWindow />
        </main>
      </ChatProvider>
    </div>
  );
}

export default function ChatApp() {
  return (
    <AuthProvider>
      <AppShell />
    </AuthProvider>
  );
}
