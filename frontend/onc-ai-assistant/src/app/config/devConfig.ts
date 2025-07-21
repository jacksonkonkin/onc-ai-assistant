// Development configuration to toggle between mock and real API
export const isDevelopmentMode = process.env.NODE_ENV === 'development';

// Toggle this to false when you want to use real backend instead of mock data
// Can also be controlled via NEXT_PUBLIC_USE_MOCK_AUTH environment variable
export const useMockAuth = process.env.NEXT_PUBLIC_USE_MOCK_AUTH === 'true' || false; // Default to false for real backend

// API base URL
export const apiBaseUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

export const devConfig = {
  useMockAuth,
  apiBaseUrl,
  isDevelopmentMode,
  
  // Mock user credentials for testing
  mockCredentials: {
    admin: { username: 'admin', password: 'any-password' },
    general: { username: 'generaluser', password: 'any-password' },
    student: { username: 'studentuser', password: 'any-password' },
    researcher: { username: 'researcheruser', password: 'any-password' },
    educator: { username: 'educatoruser', password: 'any-password' },
    policy: { username: 'policyuser', password: 'any-password' },
    indigenous: { username: 'indigenoususer', password: 'any-password' }
  }
};
