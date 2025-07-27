"use client";

import { useAuth } from "../context/AuthContext";
import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import { FaUserCircle } from "react-icons/fa";
import "./navbar.css";

export default function AccountDropdown() {
  const { user, logout } = useAuth();
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleLogout = () => {
    logout();
    // Redirect to home page after logout
    window.location.href = '/';
  };

  const handleClickOutside = (e: MouseEvent) => {
    if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
      setOpen(false);
    }
  };

  useEffect(() => {
    document.addEventListener("mousedown", handleClickOutside);
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, []);

  return (
    <div className="account-dropdown" ref={dropdownRef}>
      <button className="account-icon" onClick={() => setOpen(!open)}>
        <FaUserCircle size={28} />
      </button>
      <div className={`dropdown-menu ${open ? "open" : ""}`}>
        {user && (
          <div className="dropdown-item" style={{ fontWeight: 'bold', color: '#129DC0' }}>
            {user.username} ({user.role})
          </div>
        )}
        <Link href="/accountSettings" className="dropdown-item" onClick={() => setOpen(false)}>
          Account Settings
        </Link>
        <button className="dropdown-item" onClick={handleLogout}>
          Log out
        </button>
      </div>
    </div>
  );
}
