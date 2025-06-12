"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import "./ChatPage.css";
import { FiSend } from "react-icons/fi";

type Message = {
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
};

export default function ChatPage() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [pageLoaded, setPageLoaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Trigger fade-in after first render
    setPageLoaded(true);
  }, []);

  const fetchAIResponse = async (prompt: string): Promise<string> => {
    try {
      const response = await fetch(
        "https://onc-assistant-822f952329ee.herokuapp.com/query",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ text: prompt }),
        }
      );
      if (!response.ok) throw new Error("API request failed");
      const data = await response.json();
      return data.response ?? "No response from AI.";
    } catch (error) {
      console.error("Error fetching AI response:", error);
      return "Sorry, something went wrong.";
    }
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userMessage: Message = { sender: "user", text: input };
    setMessages((prev) => [
      ...prev,
      userMessage,
      { sender: "ai", text: "", isThinking: true },
    ]);
    setInput("");

    const aiText = await fetchAIResponse(input);

    // Stop thinking
    setMessages((prev) => {
      const updated = [...prev];
      updated[updated.length - 1] = {
        ...updated[updated.length - 1],
        isThinking: false,
      };
      return updated;
    });

    // Typewriter effect
    let index = 0;
    const typeInterval = setInterval(() => {
      setMessages((prev) => {
        const updated = [...prev];
        const currentAiMsg = updated[updated.length - 1];
        updated[updated.length - 1] = {
          ...currentAiMsg,
          text: aiText.slice(0, index + 1),
        };
        return updated;
      });

      index++;

      if (index >= aiText.length) {
        clearInterval(typeInterval);
      }
    }, 30);
  };

  useEffect(() => {
    const textarea = textareaRef.current;
    if (textarea) {
      textarea.style.height = "auto";
      textarea.style.height = textarea.scrollHeight + "px";
    }
  }, [input]);

  const renderMessageText = (msg: Message) => {
    if (msg.sender === "ai" && msg.isThinking) {
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
    <div className={`chat-container ${pageLoaded ? "fade-in" : ""}`}>
      <div className="chat-header">
        <Image
          src="/FishLogo.png"
          alt="Fish Logo"
          width={100}
          height={100}
          quality={100}
          className="fish-logo floating"
        />
        <h2>Hello! How can I assist you today?</h2>
      </div>

      <div className="chat-body">
        <div className="messages">
          {messages.map((msg, i) => (
            <div
              key={i}
              className={`message ${
                msg.sender === "user" ? "user-msg" : "ai-msg"
              }`}
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
            <FiSend size={20} color="#007acc" className="send-icon" />
          </button>
        </div>
      </div>
    </div>
  );
}
