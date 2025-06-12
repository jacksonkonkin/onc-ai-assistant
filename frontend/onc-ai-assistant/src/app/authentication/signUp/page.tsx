"use client";

import Image from "next/image";
import { useAuth } from "../../context/AuthContext";
import { useRouter } from "next/navigation";
import "../LoginPage.css";

export default function SignupPage() {
  const { setIsLoggedIn } = useAuth();
  const router = useRouter();

  const handleSignup = (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoggedIn(true);
    router.push("/chatPage");
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
        <h2 className="form-title">Sign Up</h2>
        <form onSubmit={handleSignup}>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input type="text" id="username" placeholder="Choose a username" />
          </div>
          <div className="form-group">
            <label htmlFor="email">Email</label>
            <input type="email" id="email" placeholder="Enter your email" />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Create a password"
            />
          </div>
          <div className="form-group">
            <label htmlFor="confirm-password">Confirm Password</label>
            <input
              type="password"
              id="confirm-password"
              placeholder="Re-enter your password"
            />
          </div>

          <button type="submit" className="btn-rounded-gradient">
            Sign Up
          </button>
        </form>

        <div className="signup-prompt">
          <p>Already have an account?</p>
          <a href="/authentication" className="signup-link">
            LOG IN
          </a>
        </div>
      </div>
    </div>
  );
}