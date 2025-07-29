"use client";

import React from "react";
import { FiTrash2 } from "react-icons/fi";

export type ChatHistory = {
  id: string;
  title: string;
  messages: {
    id?: string;
    sender: "user" | "ai";
    text: string;
    isThinking?: boolean;
    chat_id?: string;
    user_id?: string;
    timestamp?: string;
    rating?: number;
  }[];
};

interface ChatHistorySidebarProps {
  histories: ChatHistory[];
  selectedChatId: string;
  onSelectChat: (id: string) => void;
  onNewChat: () => void;
  onDeleteChat: (id: string) => void;
  isLoggedIn: boolean;
}

export default function ChatHistorySidebar({
  histories,
  selectedChatId,
  onSelectChat,
  onNewChat,
  onDeleteChat,
  isLoggedIn,
}: ChatHistorySidebarProps) {
  if (!isLoggedIn) return null;

  const handleDeleteChat = (e: React.MouseEvent, chatId: string) => {
    e.stopPropagation(); // Prevent selecting the chat when clicking delete
    onDeleteChat(chatId);
  };

  return (
    <div className="history-sidebar">
      {histories.map((chat) => (
        <div
          key={chat.id}
          className={`history-item ${chat.id === selectedChatId ? "active" : ""}`}
          onClick={() => onSelectChat(chat.id)}
        >
          <span className="chat-title">{chat.title}</span>
          <button
            className="delete-chat-btn"
            onClick={(e) => handleDeleteChat(e, chat.id)}
            title="Delete chat"
          >
            <FiTrash2 size={14} />
          </button>
        </div>
      ))}
      <button className="new-chat-btn" onClick={onNewChat}>
        + New Chat
      </button>
    </div>
  );
}