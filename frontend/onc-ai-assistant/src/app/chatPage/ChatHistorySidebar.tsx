"use client";

import React from "react";
import { FiTrash2 } from "react-icons/fi";

export type ChatHistory = {
  id: number;
  title: string;
  messages: { sender: "user" | "ai"; text: string; isThinking?: boolean }[];
};

interface ChatHistorySidebarProps {
  histories: ChatHistory[];
  selectedChatId: number;
  onSelectChat: (id: number) => void;
  onNewChat: () => void;
  onDeleteChat: (id: number) => void;
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

  const handleDeleteChat = (e: React.MouseEvent, chatId: number) => {
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