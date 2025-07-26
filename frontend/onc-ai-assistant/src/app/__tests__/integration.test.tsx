import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import Navbar from '../components/navbar'
import { AuthProvider } from '../context/AuthContext'

// Mock Next.js router
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

// Test component that simulates a full page with Navbar
const TestApp = ({ initialLoggedIn = false }: { initialLoggedIn?: boolean }) => {
  return (
    <AuthProvider>
      <div>
        <Navbar />
        <main>
          <h1>Test Page Content</h1>
        </main>
      </div>
    </AuthProvider>
  )
}

describe('Integration Tests', () => {
  const mockPush = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  it('renders navbar with auth context', () => {
    render(<TestApp />)
    
    // Check navbar elements
    expect(screen.getByAltText('ONC Logo')).toBeInTheDocument()
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.getByText('Admin')).toBeInTheDocument()
    
    // Check for sign in button (user not logged in by default)
    expect(screen.getByText('Sign in')).toBeInTheDocument()
  })

  it('handles navigation through navbar links', async () => {
    const user = userEvent.setup()
    render(<TestApp />)
    
    // Test navigation links
    const homeLink = screen.getByText('Home')
    const chatLink = screen.getByText('Chat')
    const adminLink = screen.getByText('Admin')
    
    expect(homeLink.closest('a')).toHaveAttribute('href', '/')
    expect(chatLink.closest('a')).toHaveAttribute('href', '/chatPage')
    expect(adminLink.closest('a')).toHaveAttribute('href', '/adminPages')
  })

  it('shows sign in button when user is not authenticated', () => {
    render(<TestApp initialLoggedIn={false} />)
    
    expect(screen.getByText('Sign in')).toBeInTheDocument()
    expect(screen.queryByRole('button', { name: /account/i })).not.toBeInTheDocument()
  })

  it('maintains consistent styling across components', () => {
    render(<TestApp />)
    
    const navbar = screen.getByRole('navigation')
    expect(navbar).toHaveClass('navBar')
    
    // Check that logo link has correct attributes
    const logoLink = screen.getByAltText('ONC Logo').closest('a')
    expect(logoLink).toHaveClass('logo')
    expect(logoLink).toHaveAttribute('href', 'https://www.oceannetworks.ca/')
  })

  it('renders all navigation sections correctly', () => {
    render(<TestApp />)
    
    // Main navigation should be present
    const navLinks = screen.getByText('Home').closest('.navLinks')
    expect(navLinks).toBeInTheDocument()
    
    // Account area should be present
    const accountArea = screen.getByText('Sign in').closest('.accountArea')
    expect(accountArea).toBeInTheDocument()
  })
})
