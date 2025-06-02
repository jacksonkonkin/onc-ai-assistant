import Link from "next/link";
import "./navbar.css";

export default function Navbar() {
  return (
    <nav className="navBar">
      <Link href="https://www.oceannetworks.ca/" className="logo">
        <img src="/ONC_Logo.png" alt="ONC Logo" className="h-20 w-auto" />
      </Link>
      <div className="navLinks">
        <Link href="/">Home</Link>
        <Link href="/chatPage">Chat</Link>
        <Link href="/adminPages">Admin</Link>
        <Link
          href="/authentication"
          className="sign-in-btn"
        >
          Sign in
        </Link>
      </div>
    </nav>
  );
}
