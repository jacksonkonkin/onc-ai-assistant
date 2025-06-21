'use client';

import './adminPanel.css';

export default function AdminPage() {

  return (
    <div className="admin-container">
      <h1>Admin Dashboard</h1>

      <main className="admin-dashboard">
        <div className="dash-column">
          <div className="module">
            <h2>View Analytics</h2>
          </div>
        </div>
        <div className="dash-column">
          <div className="module">
            <h2>Review User Feedback & Frequent Queries</h2>
          </div>

          <div className="module">
            <h2>Upload Documents</h2>
            <div className="upload"> 
              <p><i>Drag and drop/upload files here to enhance the assistant's model.</i></p>
              <label>
                Browse
                <input type="file" id="file" style={{display: 'none'}}/>
              </label>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}
