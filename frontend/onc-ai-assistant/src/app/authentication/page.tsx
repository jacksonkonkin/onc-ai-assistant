"use client";

import Image from "next/image";
import { useAuth } from "../context/AuthContext";
import { authService } from "../services/authService";
import "./LoginPage.css";
import { useRouter } from "next/navigation";
import { useState } from "react";

export default function LoginPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [formData, setFormData] = useState({
    username: "",
    password: "",
  });
  const [errors, setErrors] = useState<{[key: string]: string}>({});
  const [isLoading, setIsLoading] = useState(false);

  const handleInputChange = (field: string, value: string) => {
    setFormData(prev => ({
      ...prev,
      [field]: value
    }));
    // Clear error when user starts typing
    if (errors[field]) {
      setErrors(prev => ({
        ...prev,
        [field]: ""
      }));
    }
  };

  const handleLogin = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setErrors({});

    try {
      // Validate inputs
      const newErrors: {[key: string]: string} = {};
      if (!formData.username.trim()) {
        newErrors.username = "Username is required";
      }
      if (!formData.password.trim()) {
        newErrors.password = "Password is required";
      }
      
      if (Object.keys(newErrors).length > 0) {
        setErrors(newErrors);
        return;
      }

      // Call login API
      const response = await authService.login(formData);
      
      // Decode JWT to get user info
      const tokenData = authService.decodeToken(response.access_token);
      
      if (tokenData) {
        const userData = {
          username: tokenData.sub,
          role: tokenData.role,
          isIndigenous: false, // This info isn't in JWT, could be fetched separately if needed
        };

        // Update auth context
        login(response.access_token, userData);
        
        router.push("/chatPage");
      } else {
        throw new Error("Failed to decode authentication token");
      }
      
    } catch (error: any) {
      console.error("Login error:", error);
      setErrors({ 
        submit: error.message || "Login failed. Please check your credentials." 
      });
    } finally {
      setIsLoading(false);
    }
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
            <input 
              type="text" 
              id="username" 
              placeholder="Type your username"
              value={formData.username}
              onChange={(e) => handleInputChange('username', e.target.value)}
              className={errors.username ? 'error' : ''}
            />
            {errors.username && <span className="error-message">{errors.username}</span>}
          </div>
          <div className="form-group">
            <label htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              placeholder="Type your password"
              value={formData.password}
              onChange={(e) => handleInputChange('password', e.target.value)}
              className={errors.password ? 'error' : ''}
            />
            {errors.password && <span className="error-message">{errors.password}</span>}
          </div>

          {errors.submit && (
            <div className="error-message" style={{ 
              textAlign: 'center', 
              marginBottom: '10px',
              padding: '10px',
              backgroundColor: '#fee',
              border: '1px solid #fcc',
              borderRadius: '4px',
              color: '#c33'
            }}>
              {errors.submit}
            </div>
          )}

          <div className="form-footer">
            <a href="/authentication/forgotPassword" className="forgot-password">
              Forgot password?
            </a>
          </div>

          <button 
            type="submit" 
            className="btn-rounded-gradient"
            disabled={isLoading}
          >
            {isLoading ? "Logging in..." : "Login"}
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
