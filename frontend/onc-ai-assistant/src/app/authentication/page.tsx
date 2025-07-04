"use client";

import Image from "next/image";
import { useAuth } from "../context/AuthContext";
import "./LoginPage.css";
import { useRouter } from "next/navigation";

export default function LoginPage() {
  const { setIsLoggedIn } = useAuth(); // TEMP AUTH
  const router = useRouter();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoggedIn(true);
    router.push("/chatPage"); // Redirect after login
  };

  return (
    <div className="login-page">
      <Image
        src="/authPageArt.jpg"
        alt="Background"
        layout="fill"
        objectFit="cover"
        className="bg-image"
      />

      <div className="login-form">
        <h2 className="form-title">Login</h2>
        <form onSubmit={handleLogin}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input type="text" id="username" placeholder="Type your username" />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Type your password"
            />
          </div>

          <div className="form-footer">
            <a href="/authentication/forgotPassword" className="forgot-password">
              Forgot password?
            </a>
          </div>

          <button type="submit" className="btn-rounded-gradient">
            Login
          </button>
        </form>

        <div className="signup-prompt">
          <p>Or Sign Up Using</p>
          <a href="/authentication/signUp" className="signup-link">
            SIGN UP
          </a>
        </div>
      </div>
    </div>
  );
}
