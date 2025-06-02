"use client";

import Image from "next/image";
import "./LoginPage.css";

export default function LoginPage() {
  return (
    <div className="login-page">
      <Image src="/authPageArt.jpg" alt="Background" layout="fill" objectFit="cover" className="bg-image" />

      <div className="login-form">
        <h2 className="form-title">Login</h2>
        <form>
          <div className="form-group">
            <label htmlFor="username">Username</label>
            <input type="text" id="username" placeholder="Type your username" />
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input type="password" id="password" placeholder="Type your password" />
          </div>

          <div className="form-footer">
            <a href="#" className="forgot-password">Forgot password?</a>
          </div>

          <button type="submit" className="login-button">Login</button>
        </form>

        <div className="signup-prompt">
          <p>Or Sign Up Using</p>
          <a href="#" className="signup-link">SIGN UP</a>
        </div>
      </div>
    </div>
  );
}