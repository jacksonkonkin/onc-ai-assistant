"use client";

import { useAuth } from "../context/AuthContext";
import Link from "next/link";
import { useState, useRef, useEffect } from "react";
import "./navbar.css"; // assuming you're still using the same CSS

export default function AccountDropdown() {
  const { setIsLoggedIn } = useAuth();
  const [open, setOpen] = useState(false);
  const dropdownRef = useRef<HTMLDivElement>(null);

  const handleLogout = () => {
    setIsLoggedIn(false);
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
        👤
      </button>
      {open && (
        <div className="dropdown-menu">
          <Link href="/accountSettings" className="dropdown-item">
            Account Settings
          </Link>
          <button className="dropdown-item" onClick={handleLogout}>
            Log out
          </button>
        </div>
      )}
    </div>
  );
}
