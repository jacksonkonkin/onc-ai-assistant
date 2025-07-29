"use client";

import Link from "next/link";
import { useAuth } from "../context/AuthContext";
import AccountDropdown from "./accountDropdown";
import "./navbar.css";

export default function Navbar() {
  const { isLoggedIn, user } = useAuth();

  return (
    <nav className="navBar">
      <Link href="https://www.oceannetworks.ca/" className="logo">
        <img src="/ONC_Logo.png" alt="ONC Logo" className="h-20 w-auto" />
      </Link>
      <div className="navLinks">
        <div className="mainLinks">
          <Link href="/">Home</Link>
          <Link href="/chatPage">Chat</Link>
          {/* Only show Admin link for admin users */}
          {isLoggedIn && user?.role === 'admin' && (
            <Link href="/adminPages">Admin</Link>
          )}
        </div>

        <div className="accountArea">
          {isLoggedIn ? (
            <AccountDropdown />
          ) : (
            <Link href="/authentication" className="sign-in-btn">
              Sign in
            </Link>
          )}
        </div>
      </div>
    </nav>
  );
}
