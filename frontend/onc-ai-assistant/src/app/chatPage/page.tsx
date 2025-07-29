"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { FiSend } from "react-icons/fi";
import { useAuth } from "../context/AuthContext";
import ChatHistorySidebar from "./ChatHistorySidebar";
import ChatHistoryManager from "./ChatHistoryManager";
import "./chatPage.css";

type Message = {
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
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

  const fetchAIResponse = async (prompt: string): Promise<string> => {
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
  };

  const createHandleSend = (addMessageToChat: (message: Message) => void, updateLastMessage: (updates: Partial<Message>) => void, selectedChat: any) => async () => {
    if (!input.trim() || !selectedChat) return;

    const userMessage: Message = { sender: "user", text: input };
    addMessageToChat(userMessage);
    addMessageToChat({ sender: "ai", text: "", isThinking: true });
    
    const userInput = input;
    setInput("");

    try {
      const aiText = await fetchAIResponse(userInput);

      // Type out the AI response
      let index = 0;
      const typeInterval = setInterval(() => {
        updateLastMessage({
          isThinking: false,
          text: aiText.slice(0, index + 1),
        });

        index++;
        if (index >= aiText.length) {
          clearInterval(typeInterval);
          
          // Save AI response to database after typing animation completes
          if (isLoggedIn) {
            saveAIMessageToDatabase(aiText, selectedChat.id);
          }
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

  const saveAIMessageToDatabase = async (aiText: string, chatId: string) => {
    try {
      const response = await fetch(
        "https://onc-assistant-822f952329ee.herokuapp.com/api/messages",
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: aiText,
            chat_id: chatId,
            user_id: "-1", // Special ID to indicate AI message
          }),
        }
      );

      if (!response.ok) {
        console.error("Failed to save AI message to database");
      }
    } catch (error) {
      console.error("Error saving AI message:", error);
    }
  };

  const handleKeyPress = (handleSend: () => void) => (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
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
        const handleSend = createHandleSend(addMessageToChat, updateLastMessage, selectedChat);
        
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
                        {renderMessageText(msg)}
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
                    onKeyPress={handleKeyPress(handleSend)}
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