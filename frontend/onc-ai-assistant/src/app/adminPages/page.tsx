'use client';

import { useAuth } from '../context/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import './adminPanel.css';

export default function AdminPage() {
  const { isLoggedIn, user } = useAuth();
  const router = useRouter();

  useEffect(() => {
    // Check if user is logged in and has admin role
    if (!isLoggedIn) {
      router.push('/authentication');
      return;
    }
    
    if (!user || user.role !== 'admin') {
      // Redirect non-admin users to chat page
      router.push('/chatPage');
      return;
    }
  }, [isLoggedIn, user, router]);

  // Show loading or redirect while checking permissions
  if (!isLoggedIn || !user || user.role !== 'admin') {
    return (
      <div className="admin-container">
        <h1>Checking permissions...</h1>
      </div>
    );
  }

  return (
    <div className="admin-container">
      <h1>Admin Dashboard</h1>

      <main className="admin-dashboard">

        <div className="dash-column">
          {/* Analytics: will need to create (?) / display graphs and stats */}
          <div className="module">
            <h2>View Analytics</h2>
            <div className="analytics"> 
              {/* Will pull statistics/analytics from backend and render as graph. */}
            </div>
          </div>
        </div>

        <div className="dash-column">
          {/* User feedback module: will need to list common user queries/feedback */}
          <div className="module">
            <h2>Review User Feedback & Frequent Queries</h2>
            <div className="frequent-queries">
              {/* Will eventually be populated with backend data with a show more if overflowing. 
              List items will be buttoms to show review functionality */}
              <ul>
                <li>What is the average temperature in Cambridge Bay in July?</li>
                <li>What is the current temperature in Cambridge Bay?</li>
                <li>Is there a turbidity sensor in Cambridge Bay??</li>
              </ul>
            </div>
          </div>

          {/* Document upload */}
          <div className="module">
            <h2>Upload Documents</h2>
            <div className="upload"> 
              <p><i>Drag and drop/upload files here to enhance the assistant's model.</i></p>
              <label>
                Browse
                <input type="file" id="file" style={{display: 'none'}}/>
                {/* eventually will use hooks to store + upload files. currently just pops file picker open */}
              </label>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
