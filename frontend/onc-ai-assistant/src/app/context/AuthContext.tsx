// Authentication context with JWT token management
"use client";

import { createContext, useContext, useState, useEffect, ReactNode } from "react";

type User = {
  username: string;
  role: string;
  isIndigenous: boolean;
};

type AuthContextType = {
  isLoggedIn: boolean;
  user: User | null;
  token: string | null;
  login: (token: string, userData: User) => void;
  logout: () => void;
  setIsLoggedIn: (value: boolean) => void; // Keep for backward compatibility
};

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);

  // Load auth state from localStorage on mount
  useEffect(() => {
    const savedToken = localStorage.getItem('accessToken');
    const savedUser = localStorage.getItem('user');
    
    if (savedToken && savedUser) {
      try {
        const userData = JSON.parse(savedUser);
        setToken(savedToken);
        setUser(userData);
        setIsLoggedIn(true);
      } catch (error) {
        // Clear invalid data
        localStorage.removeItem('accessToken');
        localStorage.removeItem('user');
      }
    }
  }, []);

  const login = (accessToken: string, userData: User) => {
    setToken(accessToken);
    setUser(userData);
    setIsLoggedIn(true);
    localStorage.setItem('accessToken', accessToken);
    localStorage.setItem('user', JSON.stringify(userData));
  };

  const logout = () => {
    setToken(null);
    setUser(null);
    setIsLoggedIn(false);
    localStorage.removeItem('accessToken');
    localStorage.removeItem('user');
  };

  // Backward compatibility function
  const setIsLoggedInCompat = (value: boolean) => {
    if (!value) {
      logout();
    }
    setIsLoggedIn(value);
  };

  return (
    <AuthContext.Provider value={{ 
      isLoggedIn, 
      user, 
      token, 
      login, 
      logout, 
      setIsLoggedIn: setIsLoggedInCompat 
    }}>
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used within AuthProvider");
  return context;
}
