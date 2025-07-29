"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { FiSend, FiThumbsUp, FiThumbsDown, FiDownload } from "react-icons/fi";
import ReactMarkdown from "react-markdown";
import { useAuth } from "../context/AuthContext";
import ChatHistorySidebar from "./ChatHistorySidebar";
import ChatHistoryManager from "./ChatHistoryManager";
import "./chatPage.css";

type DownloadFile = {
  filename: string;
  download_url: string;
  size: number;
};

type Message = {
  id?: string;
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
  userQuery?: string; // Store the original user query for feedback
  feedback?: "thumbs_up" | "thumbs_down" | null;
  downloads?: DownloadFile[];
  chat_id?: string;
  user_id?: string;
  timestamp?: string;
  rating?: number;
};

export default function ChatPage() {
  const { isLoggedIn } = useAuth();
  const [input, setInput] = useState("");
  const [pageLoaded, setPageLoaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Trigger fade-in after first render
    setPageLoaded(true);
  }, []);

  const fetchAIResponse = async (prompt: string): Promise<{text: string, downloads?: DownloadFile[]}> => {
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
      return {
        text: data.response ?? "No response from AI.",
        downloads: data.downloads
      };
    } catch (error) {
      console.error("Error fetching AI response:", error);
      return { text: "Sorry, something went wrong." };
    }
  };

  const createHandleSend = (addMessageToChat: (message: Message) => void, updateLastMessage: (updates: Partial<Message>) => void) => async () => {
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
    
    addMessageToChat(userMessage);
    addMessageToChat(aiMessage);
    setInput("");

    try {
      const aiResponse = await fetchAIResponse(userQuery);

      // Typewriter effect
      let index = 0;
      const typeInterval = setInterval(() => {
        updateLastMessage({
          isThinking: false,
          text: aiResponse.text.slice(0, index + 1),
          downloads: aiResponse.downloads
        });

        index++;
        if (index >= aiResponse.text.length) {
          clearInterval(typeInterval);
        }
      }, 10);
    } catch (error) {
      console.error("Error generating AI response:", error);
      updateLastMessage({
        isThinking: false,
        text: "Sorry, something went wrong.",
      });
    }
  };


  const [messageFeedback, setMessageFeedback] = useState<Record<string, "thumbs_up" | "thumbs_down">>({});

  const submitFeedback = async (messageId: string | undefined, rating: "thumbs_up" | "thumbs_down", messages: Message[]) => {
    if (!messageId) return;
    
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
        // Update the local feedback state
        setMessageFeedback(prev => ({
          ...prev,
          [messageId]: rating
        }));
      } else {
        console.error("Failed to submit feedback");
      }
    } catch (error) {
      console.error("Error submitting feedback:", error);
    }
  };

  const handleKeyDown = (handleSend: () => void) => (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault(); // Prevent new line
      handleSend();
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
    <ChatHistoryManager>
      {({
        chatHistories,
        selectedChatId,
        selectedChat,
        messages,
        handleNewChat,
        handleDeleteChat,
        handleSelectChat,
        addMessageToChat,
        updateLastMessage,
        isLoading,
      }) => {
        const handleSend = createHandleSend(addMessageToChat, updateLastMessage);
        
        return (
          <div className="chat-page-wrapper">
            <ChatHistorySidebar
              histories={chatHistories}
              selectedChatId={selectedChatId}
              onSelectChat={handleSelectChat}
              onNewChat={handleNewChat}
              onDeleteChat={handleDeleteChat}
              isLoggedIn={isLoggedIn}
            />
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
                  {isLoading ? (
                    <div className="loading-message">Loading chat history...</div>
                  ) : (
                    messages.map((msg: Message, i: number) => (
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
                          <>
                            <div className="feedback-buttons">
                              <button
                                onClick={() => submitFeedback(msg.id, "thumbs_up", messages)}
                                className={`feedback-btn ${messageFeedback[msg.id || ''] === "thumbs_up" ? "active" : ""}`}
                                disabled={messageFeedback[msg.id || ''] !== undefined}
                                title="Helpful"
                              >
                                <FiThumbsUp size={16} />
                              </button>
                              <button
                                onClick={() => submitFeedback(msg.id, "thumbs_down", messages)}
                                className={`feedback-btn ${messageFeedback[msg.id || ''] === "thumbs_down" ? "active" : ""}`}
                                disabled={messageFeedback[msg.id || ''] !== undefined}
                                title="Not helpful"
                              >
                                <FiThumbsDown size={16} />
                              </button>
                              {messageFeedback[msg.id || ''] && (
                                <span className="feedback-thanks">
                                  Thanks for your feedback!
                                </span>
                              )}
                            </div>
                            {msg.downloads && msg.downloads.length > 0 && (
                              <div className="download-section">
                                <h4 className="download-title">Available Downloads:</h4>
                                <div className="download-buttons">
                                  {msg.downloads.map((file, index) => (
                                    <a
                                      key={index}
                                      href={`http://localhost:8000${file.download_url}`}
                                      download={file.filename}
                                      className="download-btn"
                                      title={`Download ${file.filename} (${(file.size / 1024 / 1024).toFixed(2)} MB)`}
                                    >
                                      <FiDownload size={16} />
                                      <span className="download-filename">{file.filename}</span>
                                      <span className="download-size">
                                        ({(file.size / 1024 / 1024).toFixed(2)} MB)
                                      </span>
                                    </a>
                                  ))}
                                </div>
                              </div>
                            )}
                          </>
                        )}
                      </div>
                    ))
                  )}
                </div>
                <div className="chat-input-wrapper">
                  <textarea
                    ref={textareaRef}
                    placeholder="Type a message..."
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={handleKeyDown(handleSend)}
                    className="chat-input"
                    rows={1}
                    disabled={isLoading}
                  />
                  <button 
                    onClick={handleSend} 
                    className="send-button"
                    disabled={isLoading || !input.trim()}
                  >
                    <FiSend size={20} color="#007acc" className="send-icon" />
                  </button>
                </div>
              </div>
            </div>
          </div>
        );
      }}
    </ChatHistoryManager>
  );
}