import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import Home from '../page'

// Mock Next.js router
jest.mock('next/navigation', () => ({
  useRouter: jest.fn(),
}))

// Mock Next.js Image component
jest.mock('next/image', () => ({
  __esModule: true,
  default: ({ src, alt, ...props }: any) => {
    return <img src={src} alt={alt} {...props} />
  },
}))

describe('Home (Landing Page)', () => {
  const mockPush = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  it('renders the main heading', () => {
    render(<Home />)
    
    const heading = screen.getByRole('heading', { level: 1 })
    expect(heading).toBeInTheDocument()
  })

  it('renders the descriptive text', () => {
    render(<Home />)
    
    expect(screen.getByText(/AI-powered virtual assistant/)).toBeInTheDocument()
    expect(screen.getByText(/Log in or create an account to get started/)).toBeInTheDocument()
  })

  it('renders the background image', () => {
    render(<Home />)
    
    const backgroundImage = screen.getByAltText('Background')
    expect(backgroundImage).toBeInTheDocument()
    expect(backgroundImage).toHaveAttribute('src', '/landingPageBG.jpg')
  })

  it('has a link to Oceans 3.0 Data Portal', () => {
    render(<Home />)
    
    const dataPortalLink = screen.getByText('Oceans 3.0 Data Portal')
    expect(dataPortalLink).toBeInTheDocument()
    expect(dataPortalLink.closest('a')).toHaveAttribute('href', 'https://data.oceannetworks.ca/')
  })

  it('renders Log In and Sign Up buttons', () => {
    render(<Home />)
    
    expect(screen.getByText('Log In')).toBeInTheDocument()
    expect(screen.getByText('Sign Up')).toBeInTheDocument()
  })

  it('navigates to authentication page when Log In button is clicked', async () => {
    const user = userEvent.setup()
    render(<Home />)
    
    const loginButton = screen.getByText('Log In')
    await user.click(loginButton)
    
    expect(mockPush).toHaveBeenCalledWith('/authentication')
  })

  it('navigates to sign up page when Sign Up button is clicked', async () => {
    const user = userEvent.setup()
    render(<Home />)
    
    const signUpButton = screen.getByText('Sign Up')
    await user.click(signUpButton)
    
    expect(mockPush).toHaveBeenCalledWith('/authentication/signUp')
  })

  it('displays territorial acknowledgment', () => {
    render(<Home />)
    
    const acknowledgment = screen.getByText(/We acknowledge with respect that the Cambridge Bay coastal community observatory/)
    expect(acknowledgment).toBeInTheDocument()
    expect(acknowledgment).toHaveClass('territorial-ack')
  })

  it('has correct CSS classes applied', () => {
    render(<Home />)
    
    const container = screen.getByRole('heading', { level: 1 }).closest('.landing-container')
    expect(container).toBeInTheDocument()
    
    const hero = screen.getByRole('heading', { level: 1 }).closest('.landing-hero')
    expect(hero).toBeInTheDocument()
    
    const buttonsContainer = screen.getByText('Log In').closest('.landing-buttons')
    expect(buttonsContainer).toBeInTheDocument()
  })

  it('buttons have correct CSS classes', () => {
    render(<Home />)
    
    const loginButton = screen.getByText('Log In')
    const signUpButton = screen.getByText('Sign Up')
    
    expect(loginButton).toHaveClass('landing-ctaBtn')
    expect(signUpButton).toHaveClass('landing-ctaBtn')
  })

  it('renders all main content sections', () => {
    render(<Home />)
    
    // Check for main sections
    expect(screen.getByRole('main')).toBeInTheDocument()
    expect(screen.getByRole('main')).toHaveClass('landing-hero')
    
    // Check heading structure
    expect(screen.getByRole('heading', { level: 1 })).toBeInTheDocument()
  })
})
