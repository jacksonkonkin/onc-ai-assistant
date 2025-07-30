'use client';

import { useAuth } from '../context/AuthContext';
import { useRouter } from 'next/navigation';
import { useEffect } from 'react';
import './adminPanel.css';
import DocUpload from './docUpload';
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
          <Analytics />
        </div>

        <div className="dash-column">
          <ReviewQueries />

          <DocUpload />
        </div>
      </main>
    </div>
  );
}
