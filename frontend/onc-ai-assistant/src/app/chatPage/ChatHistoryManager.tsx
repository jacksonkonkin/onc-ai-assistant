"use client";

import { useState, useEffect } from "react";
import { useAuth } from "../context/AuthContext";
import { ChatHistory } from "./ChatHistorySidebar";

type Message = {
  id?: string;
  sender: "user" | "ai";
  text: string;
  isThinking?: boolean;
  chat_id?: string;
  user_id?: string;
  timestamp?: string;
  rating?: number;
};

type ApiChatHistory = {
  id: string;
  summary: string;
  user_id: string;
  last_timestamp: string;
};

type ApiMessage = {
  id: string;
  text: string;
  chat_id: string;
  user_id: string;
  timestamp: string;
  rating?: number;
};

interface ChatHistoryManagerProps {
  children: (props: {
    chatHistories: ChatHistory[];
    selectedChatId: string;
    selectedChat: ChatHistory | undefined;
    messages: Message[];
    handleNewChat: () => void;
    handleDeleteChat: (chatId: string) => void;
    handleSelectChat: (chatId: string) => void;
    addMessageToChat: (message: Message) => void;
    updateLastMessage: (updates: Partial<Message>) => void;
    isLoading: boolean;
  }) => React.ReactNode;
}

export default function ChatHistoryManager({ children }: ChatHistoryManagerProps) {
  const { isLoggedIn, user, token } = useAuth();
  const [chatHistories, setChatHistories] = useState<ChatHistory[]>([]);
  const [selectedChatId, setSelectedChatId] = useState<string>("");
  const [isLoading, setIsLoading] = useState(true);
  const [isInitialized, setIsInitialized] = useState(false);

  const API_BASE_URL = "https://onc-assistant-822f952329ee.herokuapp.com";
  
  // Use real user ID when available, fall back to username or temp ID
  const userId = user?.username || "temp-user-id"; 

  // Convert API chat to our ChatHistory format
  const convertApiChatToHistory = (apiChat: ApiChatHistory, messages: ApiMessage[] = []): ChatHistory => ({
    id: apiChat.id,
    title: apiChat.summary || "New Chat",
    messages: messages.map(msg => ({
      id: msg.id,
      sender: msg.user_id === "-1" ? "ai" : (msg.user_id === userId ? "user" : "ai"), // -1 = AI, matching userId = user, other = ai
      text: msg.text,
      chat_id: msg.chat_id,
      user_id: msg.user_id,
      timestamp: msg.timestamp,
      rating: msg.rating,
    }))
  });

  // Helper function to get headers with auth token
  const getAuthHeaders = () => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json',
    };
    if (token) {
      headers['Authorization'] = `Bearer ${token}`;
    }
    return headers;
  };

  // Load chat histories from API
  const loadChatHistories = async () => {
    if (isInitialized) return;
    
    if (!isLoggedIn) {
      // No chat histories for non-logged in users
      setChatHistories([]);
      setSelectedChatId("");
      setIsLoading(false);
      setIsInitialized(true);
      return;
    }

    try {
      setIsLoading(true);
      const response = await fetch(`${API_BASE_URL}/api/chat-histories?user_id=${userId}`, {
        headers: getAuthHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to load chat histories');
      }

      const apiChats: ApiChatHistory[] = await response.json();
      console.warn("Loaded chat histories:", apiChats);
      
      // Convert API chats to our format (without messages for now)
      const histories = apiChats.map(chat => convertApiChatToHistory(chat));
      console.warn("Converted chat histories:", histories);

      if (histories.length > 0) {
        setChatHistories(histories);
        setSelectedChatId(histories[0].id);
      } else {
        // Create a new chat directly instead of calling createNewChat() to avoid double creation
        const newChatTitle = user ? `${user.username}'s New Chat` : "New Chat";
        const newId = Date.now().toString();
        const newChat: ChatHistory = { 
          id: newId, 
          title: newChatTitle, 
          messages: [] 
        };
        setChatHistories([newChat]);
        setSelectedChatId(newId);
      }
      
    } catch (error) {
      console.error('Failed to load chat histories:', error);
      // Fallback: create a new chat only
      const newId = Date.now().toString();
      const newChat: ChatHistory = { 
        id: newId, 
        title: "New Chat", 
        messages: [] 
      };
      setChatHistories([newChat]);
      setSelectedChatId(newId);
    } finally {
      setIsLoading(false);
      setIsInitialized(true);
    }
  };

  // Load messages for a specific chat
  const loadMessages = async (chatId: string) => {
    try {
      const response = await fetch(`${API_BASE_URL}/api/messages?chat_id=${chatId}`, {
        headers: getAuthHeaders(),
      });
      
      if (!response.ok) {
        throw new Error('Failed to load messages');
      }

      const apiMessages: ApiMessage[] = await response.json();
      
      // Update the chat with loaded messages
      setChatHistories(prev => 
        prev.map(chat => 
          chat.id === chatId 
            ? {
                ...chat,
                messages: apiMessages.map(msg => ({
                  id: msg.id,
                  sender: msg.user_id === "-1" ? "ai" : (msg.user_id === userId ? "user" : "ai"), // -1 = AI, matching userId = user, other = ai
                  text: msg.text,
                  chat_id: msg.chat_id,
                  user_id: msg.user_id,
                  timestamp: msg.timestamp,
                  rating: msg.rating,
                }))
              }
            : chat
        )
      );
    } catch (error) {
      console.error('Failed to load messages:', error);
    }
  };

  // Create new chat via API
  const createNewChat = async () => {
    const newChatTitle = user ? `${user.username}'s New Chat` : "New Chat";
    
    if (!isLoggedIn) {
      // Fallback for non-logged in users
      const newId = Date.now().toString();
      const newChat: ChatHistory = { 
        id: newId, 
        title: "New Chat", 
        messages: [] 
      };
      setChatHistories(prev => [newChat, ...prev.filter(chat => chat.id !== newId)]);
      setSelectedChatId(newId);
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat-histories`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          summary: newChatTitle,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to create new chat');
      }

      const newApiChat: ApiChatHistory = await response.json();
      const newChat = convertApiChatToHistory(newApiChat);
      
      setChatHistories(prev => [newChat, ...prev]);
      setSelectedChatId(newChat.id);
      
    } catch (error) {
      console.error('Failed to create new chat:', error);
      // Fallback to local state
      const newId = Date.now().toString();
      const newChat: ChatHistory = { 
        id: newId, 
        title: newChatTitle, 
        messages: [] 
      };
      setChatHistories(prev => [newChat, ...prev.filter(chat => chat.id !== newId)]);
      setSelectedChatId(newId);
    }
  };

  useEffect(() => {
    if (!isInitialized) {
      loadChatHistories();
    }
  }, [isLoggedIn, user?.username]);

  const selectedChat = chatHistories.find((c) => c.id === selectedChatId);
  const messages = selectedChat ? selectedChat.messages : [];

  const handleNewChat = async () => {
    await createNewChat();
  };

  const handleDeleteChat = async (chatId: string) => {
    // Prevent deletion if this is the only chat left
    if (chatHistories.length <= 1) {
      console.log("Cannot delete the last remaining chat");
      return;
    }

    if (!isLoggedIn) {
      // Local deletion for non-logged in users
      setChatHistories((prev) => {
        const filteredHistories = prev.filter((chat) => chat.id !== chatId);
        
        if (chatId === selectedChatId) {
          if (filteredHistories.length > 0) {
            setSelectedChatId(filteredHistories[0].id);
          }
        }
        
        return filteredHistories;
      });
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/chat-histories/${chatId}`, {
        method: 'DELETE',
        headers: getAuthHeaders(),
      });

      if (!response.ok) {
        throw new Error('Failed to delete chat');
      }

      // Remove from local state
      setChatHistories((prev) => {
        const filteredHistories = prev.filter((chat) => chat.id !== chatId);
        
        if (chatId === selectedChatId) {
          if (filteredHistories.length > 0) {
            setSelectedChatId(filteredHistories[0].id);
          }
          // No need for else case since we guarantee at least one chat remains
        }
        
        return filteredHistories;
      });
    } catch (error) {
      console.error('Failed to delete chat:', error);
      // Fallback to local deletion even on API error
      setChatHistories((prev) => {
        const filteredHistories = prev.filter((chat) => chat.id !== chatId);
        
        if (chatId === selectedChatId) {
          if (filteredHistories.length > 0) {
            setSelectedChatId(filteredHistories[0].id);
          }
          // No need for else case since we guarantee at least one chat remains
        }
        
        return filteredHistories;
      });
    }
  };

  const handleSelectChat = async (chatId: string) => {
    setSelectedChatId(chatId);
    
    // Don't try to load messages if not logged in
    if (!isLoggedIn) {
      return;
    }
    
    // Load messages for this chat if not already loaded
    const chat = chatHistories.find(c => c.id === chatId);
    if (chat && chat.messages.length === 0) {
      await loadMessages(chatId);
    }
  };

  const addMessageToChat = async (message: Message) => {
    // Add message to local state immediately for UI responsiveness
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

    // Don't save to API if user is not logged in or if it's a thinking message
    if (!isLoggedIn || message.isThinking) {
      return;
    }

    try {
      const response = await fetch(`${API_BASE_URL}/api/messages`, {
        method: 'POST',
        headers: getAuthHeaders(),
        body: JSON.stringify({
          text: message.text,
          chat_id: selectedChatId,
          user_id: userId,
        }),
      });

      if (!response.ok) {
        throw new Error('Failed to save message');
      }

      const savedMessage: ApiMessage = await response.json();
      
      // Update the message with the API response (including ID)
      setChatHistories((prev) =>
        prev.map((chat) => {
          if (chat.id !== selectedChatId) return chat;
          const updatedMessages = [...chat.messages];
          const lastIndex = updatedMessages.length - 1;
          updatedMessages[lastIndex] = {
            ...updatedMessages[lastIndex],
            id: savedMessage.id,
            timestamp: savedMessage.timestamp,
          };
          return { ...chat, messages: updatedMessages };
        })
      );
    } catch (error) {
      console.error('Failed to save message:', error);
      // Message is already added to local state, so we can continue without API
    }
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
        isLoading,
      })}
    </>
  );
}
