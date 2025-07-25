"use client";

import { useState, useEffect } from "react";
import { ChatHistory } from "./ChatHistorySidebar";

type Message = {
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
};

interface ChatHistoryManagerProps {
  children: (props: {
    chatHistories: ChatHistory[];
    selectedChatId: number;
    selectedChat: ChatHistory | undefined;
    messages: Message[];
    handleNewChat: () => void;
    handleDeleteChat: (chatId: number) => void;
    handleSelectChat: (chatId: number) => void;
    addMessageToChat: (message: Message) => void;
    updateLastMessage: (updates: Partial<Message>) => void;
  }) => React.ReactNode;
}

export default function ChatHistoryManager({ children }: ChatHistoryManagerProps) {
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
  }, []);

  const selectedChat = chatHistories.find((c) => c.id === selectedChatId);
  const messages = selectedChat ? selectedChat.messages : [];

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

  const handleSelectChat = (chatId: number) => {
    setSelectedChatId(chatId);
  };

  const addMessageToChat = (message: Message) => {
    setChatHistories((prev) =>
      prev.map((chat) =>
        chat.id === selectedChatId
          ? {
              ...chat,
              messages: [...chat.messages, message],
            }
          : chat
      )
    );
  };

  const updateLastMessage = (updates: Partial<Message>) => {
    setChatHistories((prev) =>
      prev.map((chat) => {
        if (chat.id !== selectedChatId) return chat;
        const updatedMessages = [...chat.messages];
        const lastMsg = updatedMessages[updatedMessages.length - 1];
        updatedMessages[updatedMessages.length - 1] = {
          ...lastMsg,
          ...updates,
        };
        return { ...chat, messages: updatedMessages };
      })
    );
  };

  return (
    <>
      {children({
        chatHistories,
        selectedChatId,
        selectedChat,
        messages,
        handleNewChat,
        handleDeleteChat,
        handleSelectChat,
        addMessageToChat,
        updateLastMessage,
      })}
    </>
  );
}
