import Link from "next/link";

export default function Navbar() {
  return (
    <nav className="navBar">
      <h1>ONC AI Assistant</h1>
      <div>
        <Link href="/">Home</Link>
        <Link href="/chatPage">Chat</Link>
        <Link href="/adminPages">Admin</Link>
      </div>
    </nav>
  );
}