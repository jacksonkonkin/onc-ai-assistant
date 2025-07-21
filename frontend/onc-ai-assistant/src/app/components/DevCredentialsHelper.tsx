// Development helper component to show mock credentials
"use client";

import { devConfig } from '../config/devConfig';

export function DevCredentialsHelper() {
  // Only show in development mode with mock auth enabled
  if (!devConfig.isDevelopmentMode || !devConfig.useMockAuth) {
    return null;
  }

  return (
    <div style={{
      position: 'fixed',
      top: '10px',
      right: '10px',
      background: '#f0f8ff',
      border: '2px solid #007bff',
      borderRadius: '8px',
      padding: '15px',
      fontSize: '12px',
      maxWidth: '250px',
      zIndex: 1000,
      boxShadow: '0 2px 10px rgba(0,0,0,0.1)'
    }}>
      <h4 style={{ margin: '0 0 10px 0', color: '#007bff', fontSize: '14px' }}>
        🛠️ Development Mode
      </h4>
      <p style={{ margin: '0 0 8px 0', fontWeight: 'bold' }}>
        Mock Auth Enabled - Use any password
      </p>
      <div style={{ fontSize: '11px', lineHeight: '1.4' }}>
        <strong>Test Usernames:</strong>
        <ul style={{ margin: '5px 0', paddingLeft: '15px' }}>
          <li><code>admin</code> - Admin role</li>
          <li><code>generaluser</code> - General role</li>
          <li><code>studentuser</code> - Student role</li>
          <li><code>researcheruser</code> - Researcher role</li>
          <li><code>educatoruser</code> - Educator role</li>
          <li><code>policyuser</code> - Policy maker role</li>
        </ul>
        <p style={{ margin: '5px 0', fontSize: '10px', color: '#666' }}>
          Toggle mock mode in /config/devConfig.ts
        </p>
      </div>
    </div>
  );
}
