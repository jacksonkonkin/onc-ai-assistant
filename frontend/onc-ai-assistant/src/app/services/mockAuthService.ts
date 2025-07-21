// Mock authentication service for frontend development without backend
export const mockAuthService = {
  // Mock login - accepts any credentials and returns a fake JWT
  async login(credentials: { username: string; password: string }) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 500));
    
    // Mock different user types based on username
    const userRoles = {
      'admin': 'admin',
      'generaluser': 'general', 
      'studentuser': 'student',
      'researcheruser': 'researcher',
      'educatoruser': 'educator',
      'policyuser': 'policy-maker',
      'indigenoususer': 'general'
    };
    
    const role = userRoles[credentials.username as keyof typeof userRoles] || 'general';
    
    // Create a fake JWT token (not real encryption, just for development)
    const fakeToken = btoa(JSON.stringify({
      sub: credentials.username,
      role: role,
      exp: Date.now() + 24 * 60 * 60 * 1000 // 24 hours from now
    }));

    return {
      access_token: fakeToken,
      token_type: "bearer"
    };
  },

  // Mock signup - always succeeds
  async signup(userData: any) {
    // Simulate API delay
    await new Promise(resolve => setTimeout(resolve, 1000));
    
    return {
      message: "Signup successful",
      id: "mock-user-id-" + Date.now()
    };
  },

  // Decode the fake token
  decodeToken(token: string) {
    try {
      return JSON.parse(atob(token));
    } catch {
      return null;
    }
  }
};

// Mock API endpoints for testing
export const mockApi = {
  async getUsers() {
    await new Promise(resolve => setTimeout(resolve, 300));
    return {
      users: [
        { id: "1", username: "admin", role: "admin", email: "admin@test.com" },
        { id: "2", username: "generaluser", role: "general", email: "user@test.com" }
      ],
      total: 2
    };
  },

  async getAdminStats() {
    await new Promise(resolve => setTimeout(resolve, 300));
    return {
      total_users: 10,
      admin_users: 1,
      regular_users: 9,
      role_breakdown: {
        admin: 1,
        general: 5,
        student: 2,
        researcher: 1,
        educator: 1
      }
    };
  }
};
