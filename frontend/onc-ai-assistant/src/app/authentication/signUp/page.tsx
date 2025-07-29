"use client";

import Image from "next/image";
import { useAuth } from "../../context/AuthContext";
import { useRouter } from "next/navigation";
import { useState } from "react";
import { authService } from "../../services/authService";
import "../LoginPage.css";

interface FormData {
  username: string;
  email: string;
  confirmEmail: string;
  userType: string;
  indigenous: string;
  oncApiToken: string;
  password: string;
  confirmPassword: string;
}

export default function SignupPage() {
  const { login } = useAuth();
  const router = useRouter();
  const [currentStep, setCurrentStep] = useState(1);
  const [isLoading, setIsLoading] = useState(false);
  const [formData, setFormData] = useState<FormData>({
    username: "",
    email: "",
    confirmEmail: "",
    userType: "",
    indigenous: "",
    oncApiToken: "",
    password: "",
    confirmPassword: "",
  });
  const [errors, setErrors] = useState<{[key: string]: string}>({});

  const totalSteps = 4;

  const validateStep1 = () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!formData.username.trim()) {
      newErrors.username = "Username is required";
    }
    if (!formData.email.trim()) {
      newErrors.email = "Email is required";
    }
    if (!formData.confirmEmail.trim()) {
      newErrors.confirmEmail = "Please confirm your email";
    }
    if (formData.email && formData.confirmEmail && formData.email !== formData.confirmEmail) {
      newErrors.confirmEmail = "Emails do not match";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const validateStep2 = () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!formData.userType) {
      newErrors.userType = "Please select a user type";
    }
    if (!formData.indigenous) {
      newErrors.indigenous = "Please select an option";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const validateStep3 = () => {
    const newErrors: {[key: string]: string} = {};
    
    if (!formData.password.trim()) {
      newErrors.password = "Password is required";
    } else if (formData.password.length < 6) {
      newErrors.password = "Password must be at least 6 characters";
    }
    if (!formData.confirmPassword.trim()) {
      newErrors.confirmPassword = "Please confirm your password";
    }
    if (formData.password && formData.confirmPassword && formData.password !== formData.confirmPassword) {
      newErrors.confirmPassword = "Passwords do not match";
    }
    
    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  const handleInputChange = (field: keyof FormData, value: string) => {
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

  const handleNext = () => {
    if (currentStep === 1 && validateStep1()) {
      setCurrentStep(2);
    } else if (currentStep === 2 && validateStep2()) {
      setCurrentStep(3);
    } else if (currentStep === 3 && validateStep3()) {
      setCurrentStep(4);
    }
  };

  const handlePrevious = () => {
    if (currentStep > 1) {
      setCurrentStep(currentStep - 1);
    }
  };

  const handleSignup = async (e: React.FormEvent) => {
    e.preventDefault();
    
    // Only proceed with signup if we're on the final step
    if (currentStep === totalSteps) {
      setIsLoading(true);
      setErrors({});
      
      try {
        // Prepare data for backend
        const signupData = {
          username: formData.username,
          email: formData.email,
          password: formData.password,
          onc_token: formData.oncApiToken || undefined,
          is_indigenous: formData.indigenous === 'yes',
          role: formData.userType,
        };

        // Create the account
        await authService.signup(signupData);
        
        // Automatically log the user in after successful signup
        const loginResponse = await authService.login({
          username: formData.username,
          password: formData.password,
        });

        // Decode the JWT to get user info
        const tokenData = authService.decodeToken(loginResponse.access_token);
        
        if (tokenData) {
          const userData = {
            username: tokenData.sub,
            role: tokenData.role,
            isIndigenous: formData.indigenous === 'yes',
          };

          // Update auth context
          login(loginResponse.access_token, userData);
          
          router.push("/chatPage");
        } else {
          throw new Error("Failed to decode authentication token");
        }
        
      } catch (error: any) {
        console.error("Signup error:", error);
        setErrors({ 
          submit: error.message || "Signup failed. Please try again." 
        });
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    // Prevent form submission on Enter key unless we're on the final step
    if (e.key === 'Enter' && currentStep < totalSteps) {
      e.preventDefault();
      handleNext();
    }
  };

  const renderProgressBar = () => (
    <div className="progress-container">
      <div className="progress-bar">
        {[1, 2, 3, 4].map((step) => (
          <div key={step} className="progress-step-container">
            <div className={`progress-step ${currentStep >= step ? 'active' : ''} ${currentStep > step ? 'completed' : ''}`}>
              {currentStep > step ? '✓' : step}
            </div>
            {step < totalSteps && (
              <div className={`progress-line ${currentStep > step ? 'completed' : ''}`}></div>
            )}
          </div>
        ))}
      </div>
      <div className="progress-labels">
        <span className={currentStep >= 1 ? 'active-label' : ''}>Account Info</span>
        <span className={currentStep >= 2 ? 'active-label' : ''}>User Details</span>
        <span className={currentStep >= 3 ? 'active-label' : ''}>Password</span>
        <span className={currentStep >= 4 ? 'active-label' : ''}>Review</span>
      </div>
    </div>
  );

  const renderStep1 = () => (
    <div className="step-content">
      <h3>Step 1: Account Information</h3>
      <div className="form-group">
        <label htmlFor="username">Username</label>
        <input
          type="text"
          id="username"
          placeholder="Choose a username"
          value={formData.username}
          onChange={(e) => handleInputChange('username', e.target.value)}
          className={errors.username ? 'error' : ''}
        />
        {errors.username && <span className="error-message">{errors.username}</span>}
      </div>
      <div className="form-group">
        <label htmlFor="email">Email</label>
        <input
          type="email"
          id="email"
          placeholder="Enter your email"
          value={formData.email}
          onChange={(e) => handleInputChange('email', e.target.value)}
          className={errors.email ? 'error' : ''}
        />
        {errors.email && <span className="error-message">{errors.email}</span>}
      </div>
      <div className="form-group">
        <label htmlFor="confirmEmail">Confirm Email</label>
        <input
          type="email"
          id="confirmEmail"
          placeholder="Re-enter your email"
          value={formData.confirmEmail}
          onChange={(e) => handleInputChange('confirmEmail', e.target.value)}
          className={errors.confirmEmail ? 'error' : ''}
        />
        {errors.confirmEmail && <span className="error-message">{errors.confirmEmail}</span>}
      </div>
    </div>
  );

  const renderStep2 = () => (
    <div className="step-content">
      <h3>Step 2: User Details</h3>
      <div className="form-group">
        <label htmlFor="userType">User Type</label>
        <select
          id="userType"
          className={`form-select ${errors.userType ? 'error' : ''}`}
          value={formData.userType}
          onChange={(e) => handleInputChange('userType', e.target.value)}
        >
          <option value="" disabled>Select user type</option>
          <option value="general">General</option>
          <option value="student">Student</option>
          <option value="researcher">Researcher</option>
          <option value="educator">Educator</option>
          <option value="policy-maker">Policy Maker</option>
        </select>
        {errors.userType && <span className="error-message">{errors.userType}</span>}
      </div>
      <div className="form-group">
        <label htmlFor="indigenous">Do you identify as Indigenous?</label>
        <select
          id="indigenous"
          className={`form-select ${errors.indigenous ? 'error' : ''}`}
          value={formData.indigenous}
          onChange={(e) => handleInputChange('indigenous', e.target.value)}
        >
          <option value="" disabled>Please select</option>
          <option value="yes">Yes</option>
          <option value="no">No</option>
        </select>
        {errors.indigenous && <span className="error-message">{errors.indigenous}</span>}
      </div>
      <div className="form-group">
        <label htmlFor="oncApiToken">
          ONC API Token (Optional)
          <span className="tooltip">
            <span className="question-mark">?</span>
            <span className="tooltip-text">If you have your own ONC API Token, please enter it here. If you do not have one, this can be left blank</span>
          </span>
        </label>
        <input
          type="text"
          id="oncApiToken"
          placeholder="Enter your ONC API token (can be left blank)"
          value={formData.oncApiToken}
          onChange={(e) => handleInputChange('oncApiToken', e.target.value)}
        />
      </div>
    </div>
  );

  const renderStep3 = () => (
    <div className="step-content">
      <h3>Step 3: Create Password</h3>
      <div className="form-group">
        <label htmlFor="password">Password</label>
        <input
          type="password"
          id="password"
          placeholder="Create a password"
          value={formData.password}
          onChange={(e) => handleInputChange('password', e.target.value)}
          className={errors.password ? 'error' : ''}
        />
        {errors.password && <span className="error-message">{errors.password}</span>}
        <small className="helper-text">Password must be at least 6 characters long</small>
      </div>
      <div className="form-group">
        <label htmlFor="confirmPassword">Confirm Password</label>
        <input
          type="password"
          id="confirmPassword"
          placeholder="Re-enter your password"
          value={formData.confirmPassword}
          onChange={(e) => handleInputChange('confirmPassword', e.target.value)}
          className={errors.confirmPassword ? 'error' : ''}
        />
        {errors.confirmPassword && <span className="error-message">{errors.confirmPassword}</span>}
      </div>
    </div>
  );

  const renderStep4 = () => (
    <div className="step-content">
      <h3>Step 4: Review & Confirm</h3>
      <div className="review-section">
        <h4>Account Information</h4>
        <div className="review-item">
          <span className="review-label">Username:</span>
          <span className="review-value">{formData.username}</span>
        </div>
        <div className="review-item">
          <span className="review-label">Email:</span>
          <span className="review-value">{formData.email}</span>
        </div>
        
        <h4>User Details</h4>
        <div className="review-item">
          <span className="review-label">User Type:</span>
          <span className="review-value">{formData.userType}</span>
        </div>
        <div className="review-item">
          <span className="review-label">Indigenous:</span>
          <span className="review-value">{formData.indigenous === 'yes' ? 'Yes' : 'No'}</span>
        </div>
        <div className="review-item">
          <span className="review-label">ONC API Token:</span>
          <span className="review-value">{formData.oncApiToken || 'Not provided'}</span>
        </div>
        
        <h4>Security</h4>
        <div className="review-item">
          <span className="review-label">Password:</span>
          <span className="review-value">••••••••</span>
        </div>
      </div>
      <p className="confirmation-text">
        Please review your information above. Click "Complete Sign Up" to create your account.
      </p>
      {errors.submit && (
        <div className="error-message" style={{ 
          textAlign: 'center', 
          marginTop: '10px',
          padding: '10px',
          backgroundColor: '#fee',
          border: '1px solid #fcc',
          borderRadius: '4px',
          color: '#c33'
        }}>
          {errors.submit}
        </div>
      )}
    </div>
  );

  return (
    <div className="login-page">
      <Image
        src="/authPageArt.jpg"
        alt="Background"
        layout="fill"
        objectFit="cover"
        className="bg-image"
      />

      <div className="login-form multi-step-form">
        <h2 className="form-title">Sign Up</h2>
        
        {renderProgressBar()}
        
        <form onSubmit={handleSignup} onKeyDown={handleKeyDown}>
          {currentStep === 1 && renderStep1()}
          {currentStep === 2 && renderStep2()}
          {currentStep === 3 && renderStep3()}
          {currentStep === 4 && renderStep4()}

          <div className="form-buttons">
            {currentStep > 1 && (
              <button
                type="button"
                onClick={handlePrevious}
                className="btn-secondary"
              >
                Previous
              </button>
            )}
            
            {currentStep < totalSteps ? (
              <button
                type="button"
                onClick={(e) => {
                  e.preventDefault();
                  handleNext();
                }}
                className="btn-rounded-gradient"
              >
                Next
              </button>
            ) : (
              <button 
                type="submit" 
                className="btn-rounded-gradient"
                disabled={isLoading}
              >
                {isLoading ? "Creating Account..." : "Complete Sign Up"}
              </button>
            )}
          </div>
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