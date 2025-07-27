import { render, screen } from '@testing-library/react'
import { useRouter } from 'next/navigation'
import Navbar from '../navbar'
import { AuthProvider } from '../../context/AuthContext'

// Mock Next.js router
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

const renderNavbar = () => {
  return render(
    <AuthProvider>
      <Navbar />
    </AuthProvider>
  )
}

describe('Navbar', () => {
  const mockPush = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  it('renders the ONC logo', () => {
    renderNavbar()
    
    const logo = screen.getByAltText('ONC Logo')
    expect(logo).toBeInTheDocument()
    expect(logo).toHaveAttribute('src', '/ONC_Logo.png')
  })

  it('renders navigation links', () => {
    renderNavbar()
    
    expect(screen.getByText('Home')).toBeInTheDocument()
    expect(screen.getByText('Chat')).toBeInTheDocument()
    expect(screen.getByText('Admin')).toBeInTheDocument()
  })

  it('has correct navigation link URLs', () => {
    renderNavbar()
    
    const homeLink = screen.getByText('Home').closest('a')
    const chatLink = screen.getByText('Chat').closest('a')
    const adminLink = screen.getByText('Admin').closest('a')
    
    expect(homeLink).toHaveAttribute('href', '/')
    expect(chatLink).toHaveAttribute('href', '/chatPage')
    expect(adminLink).toHaveAttribute('href', '/adminPages')
  })

  it('shows sign in button when user is not logged in', () => {
    renderNavbar()
    
    const signInButton = screen.getByText('Sign in')
    expect(signInButton).toBeInTheDocument()
    expect(signInButton.closest('a')).toHaveAttribute('href', '/authentication')
  })

  it('has external link to Ocean Networks Canada website', () => {
    renderNavbar()
    
    const externalLink = screen.getByAltText('ONC Logo').closest('a')
    expect(externalLink).toHaveAttribute('href', 'https://www.oceannetworks.ca/')
  })

  it('applies correct CSS classes', () => {
    renderNavbar()
    
    const nav = screen.getByRole('navigation')
    expect(nav).toHaveClass('navBar')
  })
})
