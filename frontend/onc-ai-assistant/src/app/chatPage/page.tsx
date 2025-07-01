"use client";

import { useState, useRef, useEffect } from "react";
import Image from "next/image";
import { FiSend } from "react-icons/fi";
import { useAuth } from "../context/AuthContext";
import ChatHistorySidebar, { ChatHistory } from "./ChatHistorySidebar";
import "./ChatPage.css";

type Message = {
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
};

export default function ChatPage() {
  const { isLoggedIn } = useAuth();
  const [input, setInput] = useState("");
  const [chatHistories, setChatHistories] = useState<ChatHistory[]>([
    {
      id: 1,
      title: "Welcome",
      messages: [
        { sender: "ai", text: "Hello! How can I help you?" },
        { sender: "user", text: "Just testing the assistant." },
        { sender: "ai", text: "Great! Let me know if you have questions." },
      ],
    },
    {
      id: 2,
      title: "Ocean Data",
      messages: [
        { sender: "user", text: "Tell me about ocean networks." },
        { sender: "ai", text: "Ocean Networks Canada monitors the ocean..." },
      ],
    },
  ]);
  const [selectedChatId, setSelectedChatId] = useState<number>(0);
  const [pageLoaded, setPageLoaded] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    // Create a new chat and add it to the existing histories
    const newId = Date.now();
    const newChat: ChatHistory = { 
      id: newId, 
      title: "New Chat", 
      messages: [] 
    };
    setChatHistories(prev => [...prev, newChat]);
    setSelectedChatId(newId);
    
    // Trigger fade-in after first render
    setPageLoaded(true);
  }, []);

  const selectedChat = chatHistories.find((c) => c.id === selectedChatId);
  const messages = selectedChat ? selectedChat.messages : [];

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

  const handleSend = async () => {
    if (!input.trim() || !selectedChat) return;

    const userMessage: Message = { sender: "user", text: input };
    setChatHistories((prev) =>
      prev.map((chat) =>
        chat.id === selectedChatId
          ? {
              ...chat,
              messages: [
                ...chat.messages,
                userMessage,
                { sender: "ai", text: "", isThinking: true },
              ],
            }
          : chat
      )
    );
    setInput("");

    const aiText = await fetchAIResponse(input);

    let index = 0;
    const typeInterval = setInterval(() => {
      setChatHistories((prev) =>
        prev.map((chat) => {
          if (chat.id !== selectedChatId) return chat;
          const updatedMessages = [...chat.messages];
          const lastMsg = updatedMessages[updatedMessages.length - 1];
          updatedMessages[updatedMessages.length - 1] = {
            ...lastMsg,
            isThinking: false,
            text: aiText.slice(0, index + 1),
          };
          return { ...chat, messages: updatedMessages };
        })
      );

      index++;
      if (index >= aiText.length) {
        clearInterval(typeInterval);
      }
    }, 10);
  };

  const handleNewChat = () => {
    const newId = Date.now();
    setChatHistories((prev) => [
      ...prev,
      { id: newId, title: `New Chat`, messages: [] },
    ]);
    setSelectedChatId(newId);
  };

  const handleDeleteChat = (chatId: number) => {
    setChatHistories((prev) => {
      const filteredHistories = prev.filter((chat) => chat.id !== chatId);
      
      // If we're deleting the currently selected chat, select another one
      if (chatId === selectedChatId) {
        if (filteredHistories.length > 0) {
          setSelectedChatId(filteredHistories[0].id);
        } else {
          // If no chats left, create a new one
          const newId = Date.now();
          const newChat = { id: newId, title: `New Chat`, messages: [] };
          setSelectedChatId(newId);
          return [newChat];
        }
      }
      
      return filteredHistories;
    });
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
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
    <div className="chat-page-wrapper">
      <ChatHistorySidebar
        histories={chatHistories}
        selectedChatId={selectedChatId}
        onSelectChat={setSelectedChatId}
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
              onKeyPress={handleKeyPress}
              className="chat-input"
              rows={1}
            />
            <button onClick={handleSend} className="send-button">
              <FiSend size={20} color="#007acc" className="send-icon" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}