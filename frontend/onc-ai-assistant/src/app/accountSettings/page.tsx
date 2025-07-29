"use client";

import { useAuth } from "../context/AuthContext";
import { useRouter } from "next/navigation";
import { useState, useEffect } from "react";
import "./accountSettings.css";

export default function AccountSettingsPage() {
  const { setIsLoggedIn } = useAuth();
  const router = useRouter();

  // Mock user data - replace with actual user data from your auth context or API
  const [userInfo, setUserInfo] = useState({
    username: "john_doe",
    email: "john.doe@example.com",
    userType: "student",
    indigenous: "no"
  });

  // Track original data to detect changes
  const [originalUserInfo, setOriginalUserInfo] = useState({
    username: "john_doe",
    email: "john.doe@example.com",
    userType: "student",
    indigenous: "no"
  });

  // Track if there are unsaved changes
  const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false);

  // Check for changes whenever userInfo updates
  useEffect(() => {
    const hasChanges = JSON.stringify(userInfo) !== JSON.stringify(originalUserInfo);
    setHasUnsavedChanges(hasChanges);
  }, [userInfo, originalUserInfo]);

  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    // Handle save settings logic here
    console.log("Settings saved", userInfo);
    
    // Update original data after saving
    setOriginalUserInfo({ ...userInfo });
    setHasUnsavedChanges(false);
  };

  const handleLogout = () => {
    if (hasUnsavedChanges) {
      const confirmLogout = window.confirm(
        "Any unsaved changes will be lost. Continue to log out?"
      );
      
      if (confirmLogout) {
        setIsLoggedIn(false);
        router.push("/authentication");
      }
      // If user clicks "No", nothing happens - they stay on the page
    } else {
      // No unsaved changes, logout immediately
      setIsLoggedIn(false);
      router.push("/authentication");
    }
  };

  const handleCancelChanges = () => {
    if (hasUnsavedChanges) {
      const confirmCancel = window.confirm(
        "Are you sure you want to cancel all changes?"
      );
      
      if (confirmCancel) {
        setUserInfo({ ...originalUserInfo });
        setHasUnsavedChanges(false);
      }
    }
  };

  const handleInputChange = (field: string, value: string) => {
    setUserInfo(prev => ({
      ...prev,
      [field]: value
    }));
  };

  return (
    <div className="settings-container">
      <div className="settings-header">
        <h1>Account Settings</h1>
        <div className="header-buttons">
          <button 
            className="back-button"
            onClick={() => router.back()}
          >
            ← Back
          </button>
          <button 
            type="button" 
            className="logout-button-header"
            onClick={handleLogout}
          >
            Logout {hasUnsavedChanges && "⚠️"}
          </button>
        </div>
      </div>

      <div className="settings-form-container">
        <form onSubmit={handleSaveSettings} className="settings-form">
          <div className="settings-section">
            <h2>Personal Information</h2>
            
            <div className="form-group">
              <label htmlFor="username">Username</label>
              <input 
                type="text" 
                id="username" 
                value={userInfo.username}
                onChange={(e) => handleInputChange('username', e.target.value)}
                className="settings-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="email">Email</label>
              <input 
                type="email" 
                id="email" 
                value={userInfo.email}
                onChange={(e) => handleInputChange('email', e.target.value)}
                className="settings-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="userType">User Type</label>
              <select 
                id="userType" 
                className="settings-select" 
                value={userInfo.userType}
                onChange={(e) => handleInputChange('userType', e.target.value)}
              >
                <option value="" disabled>Select user type</option>
                <option value="general">General</option>
                <option value="student">Student</option>
                <option value="researcher">Researcher</option>
              </select>
            </div>

            <div className="form-group">
              <label htmlFor="indigenous">Do you identify as Indigenous?</label>
              <select 
                id="indigenous" 
                className="settings-select" 
                value={userInfo.indigenous}
                onChange={(e) => handleInputChange('indigenous', e.target.value)}
              >
                <option value="" disabled>Please select</option>
                <option value="yes">Yes</option>
                <option value="no">No</option>
              </select>
            </div>
          </div>

          <div className="settings-section">
            <h2>Security</h2>
            
            <div className="form-group">
              <label htmlFor="currentPassword">Current Password</label>
              <input 
                type="password" 
                id="currentPassword" 
                placeholder="Enter current password"
                className="settings-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="newPassword">New Password</label>
              <input 
                type="password" 
                id="newPassword" 
                placeholder="Enter new password"
                className="settings-input"
              />
            </div>

            <div className="form-group">
              <label htmlFor="confirmPassword">Confirm New Password</label>
              <input 
                type="password" 
                id="confirmPassword" 
                placeholder="Confirm new password"
                className="settings-input"
              />
            </div>
          </div>

          <div className="settings-actions">
            <button type="submit" className="save-button">
              Save Changes
            </button>
            <button 
              type="button" 
              className={`cancel-button ${hasUnsavedChanges ? 'has-changes' : ''}`}
              onClick={handleCancelChanges}
            >
              Cancel Changes
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}