// src/App.jsx
import React from "react";
import { Routes, Route } from "react-router-dom";

import Welcome from "./pages/Welcome";
import LoginPage from "./pages/LoginPage";
import DocHistory from "./pages/DocHistory";
import DocManager from "./pages/DocManager";
import ChatGPTUI from "./pages/ChatGPTUI";

export default function App() {
  return (
    <Routes>
      <Route path="/" element={<Welcome />} />
      <Route path="/login" element={<LoginPage />} />
      <Route path="/docmanager" element={<DocManager />} />
      <Route path="/dochistory" element={<DocHistory />} />
      <Route path="/convert" element={<ChatGPTUI />} />
    </Routes>
  );
}
