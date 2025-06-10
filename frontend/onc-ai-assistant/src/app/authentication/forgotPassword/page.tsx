"use client";

import Image from "next/image";
import "../LoginPage.css";

export default function ForgotPasswordPage() {
  const handleReset = (e: React.FormEvent) => {
    e.preventDefault();
    alert("If this were real, a reset link would be sent!");
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
        <h2 className="form-title">Forgot Password</h2>
        <form onSubmit={handleReset}>
          <div className="form-group">
            <label htmlFor="email">Email Address</label>
            <input type="email" id="email" placeholder="Enter your email" />
          </div>

          <button type="submit" className="btn-rounded-gradient">
            Send Reset Link
          </button>
        </form>

        <div className="signup-prompt">
          <p>Remember your password?</p>
          <a href="/authentication" className="signup-link">
            LOG IN
          </a>
        </div>
      </div>
    </div>
  );
}
