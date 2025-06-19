"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import "./ChatPage.css";

type Message = {
  sender: "user" | "ai";
  text: string;
};

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  const fetchAIResponse = async (prompt: string): Promise<string> => {
<<<<<<< HEAD
    try {
      const response = await fetch(
        "https://onc-assistant-822f952329ee.herokuapp.com/query",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({ text: prompt }),
        }
      );

      if (!response.ok) {
        throw new Error("API request failed");
      }

      const data = await response.json();

      return data.response ?? "No response from AI.";
    } catch (error) {
      console.error("Error fetching AI response:", error);
      return "Sorry, something went wrong.";
    }
=======
    // TODO: Replace this with actual backend call when ready
    await new Promise((r) => setTimeout(r, 5000)); // simulate delay
    return "This is a placeholder response.";
>>>>>>> 3ca2139892ffff115835c33cfed72df2ba532b97
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { sender: "user", text: input };
<<<<<<< HEAD
    setMessages((prev) => [
      ...prev,
      userMessage,
      { sender: "ai", text: "thinking" },
    ]);
=======
    setMessages((prev) => [...prev, userMessage, { sender: "ai", text: "thinking" }]);
>>>>>>> 3ca2139892ffff115835c33cfed72df2ba532b97
    setInput("");

    const aiText = await fetchAIResponse(input);
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = { sender: "ai", text: aiText };
      return updated;
    });
  };

  // Auto-expand textarea height
  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = textarea.scrollHeight + "px";
    }
  }, [input]);

  const renderMessageText = (msg: Message) => {
    if (msg.sender === "ai" && msg.text === "thinking") {
      const dots = "Generating Response...".split("");
      return (
        <span className="thinking-animation">
          {dots.map((char, index) => (
            <span
              key={index}
              className="thinking-char"
              style={{ animationDelay: `${index * 0.1}s` }}
            >
              {char}
            </span>
          ))}
        </span>
      );
    }
    return msg.text;
  };

  return (
    <div className="chat-container">
      <div className="chat-header">
        <Image
          src="/FishLogo.png"
          alt="Fish Logo"
          width={100}
          height={100}
          quality={100}
          className="fish-logo"
        />
        <h2>Hello! How can I assist you today?</h2>
      </div>
      <div className="chat-body">
        <div className="messages">
          {messages.map((msg, i) => (
            <div
              key={i}
<<<<<<< HEAD
              className={`message ${
                msg.sender === "user" ? "user-msg" : "ai-msg"
              }`}
=======
              className={`message ${msg.sender === "user" ? "user-msg" : "ai-msg"}`}
>>>>>>> 3ca2139892ffff115835c33cfed72df2ba532b97
            >
              {renderMessageText(msg)}
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
