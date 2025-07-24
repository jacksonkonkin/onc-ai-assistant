"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import "./ChatPage.css";
import { FiSend, FiThumbsUp, FiThumbsDown } from "react-icons/fi";
import ReactMarkdown from "react-markdown";

type Message = {
  id: string;
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
  userQuery?: string; // Store the original user query for feedback
  feedback?: "thumbs_up" | "thumbs_down" | null;
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
        "http://localhost:8000/query",
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
  };

  const handleSend = async () => {
    if (!input.trim()) return;

    const userQuery = input;
    const userMessage: Message = { 
      id: Date.now().toString() + "-user", 
      sender: "user", 
      text: input 
    };
    const aiMessage: Message = { 
      id: Date.now().toString() + "-ai", 
      sender: "ai", 
      text: "", 
      isThinking: true, 
      userQuery: userQuery,
      feedback: null
    };
    
    setMessages((prev) => [...prev, userMessage, aiMessage]);
    setInput("");

    const aiText = await fetchAIResponse(userQuery);

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
    }, 10);
  };

  const submitFeedback = async (messageId: string, rating: "thumbs_up" | "thumbs_down") => {
    const message = messages.find(msg => msg.id === messageId);
    if (!message || message.sender !== "ai") return;

    try {
      const response = await fetch("http://localhost:8000/feedback", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          rating: rating,
          query: message.userQuery || "",
          response: message.text,
          metadata: {
            messageId: messageId,
            timestamp: new Date().toISOString()
          }
        }),
      });

      if (response.ok) {
        // Update the message to show feedback was submitted
        setMessages(prev => prev.map(msg => 
          msg.id === messageId 
            ? { ...msg, feedback: rating }
            : msg
        ));
      } else {
        console.error("Failed to submit feedback");
      }
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
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
    
    // Render markdown for AI responses, plain text for user messages
    if (msg.sender === "ai" && msg.text) {
      return (
        <div className="markdown-content">
          <ReactMarkdown 
            components={{
              // Custom components for better styling
              h1: ({ children }) => <h1 className="markdown-h1">{children}</h1>,
              h2: ({ children }) => <h2 className="markdown-h2">{children}</h2>,
              h3: ({ children }) => <h3 className="markdown-h3">{children}</h3>,
              p: ({ children }) => <p className="markdown-p">{children}</p>,
              ul: ({ children }) => <ul className="markdown-ul">{children}</ul>,
              ol: ({ children }) => <ol className="markdown-ol">{children}</ol>,
              li: ({ children }) => <li className="markdown-li">{children}</li>,
              strong: ({ children }) => <strong className="markdown-strong">{children}</strong>,
              em: ({ children }) => <em className="markdown-em">{children}</em>,
              code: ({ children }) => <code className="markdown-code">{children}</code>,
              blockquote: ({ children }) => <blockquote className="markdown-blockquote">{children}</blockquote>,
            }}
          >
            {msg.text}
          </ReactMarkdown>
        </div>
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
              <div className="message-content">
                {renderMessageText(msg)}
              </div>
              {msg.sender === "ai" && !msg.isThinking && msg.text && (
                <div className="feedback-buttons">
                  <button
                    onClick={() => submitFeedback(msg.id, "thumbs_up")}
                    className={`feedback-btn ${msg.feedback === "thumbs_up" ? "active" : ""}`}
                    disabled={msg.feedback !== null}
                    title="Helpful"
                  >
                    <FiThumbsUp size={16} />
                  </button>
                  <button
                    onClick={() => submitFeedback(msg.id, "thumbs_down")}
                    className={`feedback-btn ${msg.feedback === "thumbs_down" ? "active" : ""}`}
                    disabled={msg.feedback !== null}
                    title="Not helpful"
                  >
                    <FiThumbsDown size={16} />
                  </button>
                  {msg.feedback && (
                    <span className="feedback-thanks">
                      Thanks for your feedback!
                    </span>
                  )}
                </div>
              )}
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
