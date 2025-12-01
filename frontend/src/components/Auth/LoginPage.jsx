import React, { useState } from "react";
import "./LoginPage.css";
import { useAuth } from "../../context/AuthContext.jsx";
import Button from "../Common/Button.jsx";

function LoginPage() {
  const { login } = useAuth();
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");

  function handleSubmit(e) {
    e.preventDefault();
    if (!name.trim()) return;
    login({ name: name.trim(), email: email.trim() });
  }

  return (
    <div className="auth-wrapper">
      <div className="auth-card">
        <div className="auth-header">
          <div className="logo-circle">AI</div>
          <h1>Sign in</h1>
          <p>Use a demo account to access your ChatGPT-style interface.</p>
        </div>
        <form onSubmit={handleSubmit} className="auth-form">
          <label>
            Name
            <input
              type="text"
              placeholder="chirag"
              value={name}
              onChange={(e) => setName(e.target.value)}
              required
            />
          </label>
          <label>
            Email (optional)
            <input
              type="email"
              placeholder="chirag@kirankumari.com"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </label>
          <Button type="submit" variant="primary" className="auth-submit">
            Continue
          </Button>
        </form>
        <p className="auth-footer">
          
        </p>
      </div>
    </div>
  );
}

export default LoginPage;
