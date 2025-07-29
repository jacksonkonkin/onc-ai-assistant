'use client';

import { useAuth } from '../context/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import './adminPanel.css';
import ReviewQueries from './reviewQueries';
import Analytics from './analytics';

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
          <Analytics />
        </div>

        <div className="dash-column">
          <ReviewQueries />

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
