"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import "./ChatPage.css";

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<string[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages([...messages, input]);
    setInput("");
  };

  // Auto-expand textarea height
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = textarea.scrollHeight + "px";
    }
  }, [input]);

  return (
    <div className="chat-container">
      <div className="chat-header">
        <Image
          src="/FishLogo.png"
          alt="Fish Logo"
          width={100}
          height={100}
          className="fish-logo"
        />
        <h2>Hello! How can I assist you today?</h2>
      </div>
      <div className="chat-body">
        <div className="messages">
          {messages.map((msg, i) => (
            <div key={i} className="message">
              {msg}
            </div>
          ))}
        </div>
        <div className="chat-input-wrapper">
          <textarea
            ref={textareaRef}
            placeholder="Type a message..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
            className="chat-input"
            rows={1}
          />
          <button onClick={handleSend} className="send-button">
            <svg
              xmlns="http://www.w3.org/2000/svg"
              className="send-icon"
              height="20"
              viewBox="0 0 24 24"
              width="20"
            >
              <path d="M2 21l21-9L2 3v7l15 2-15 2v7z" />
            </svg>
          </button>
        </div>
      </div>
    </div>
  );
}
