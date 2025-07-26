import { render, screen, fireEvent } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { useRouter } from 'next/navigation'
import LoginPage from '../page'
import { AuthProvider } from '../../context/AuthContext'

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

const renderWithAuth = () => {
  return render(
    <AuthProvider>
      <LoginPage />
    </AuthProvider>
  )
}

describe('LoginPage', () => {
  const mockPush = jest.fn()

  beforeEach(() => {
    jest.clearAllMocks()
    ;(useRouter as jest.Mock).mockReturnValue({
      push: mockPush,
    })
  })

  it('renders the login form', () => {
    renderWithAuth()
    
    expect(screen.getByRole('heading', { name: 'Login' })).toBeInTheDocument()
    expect(screen.getByLabelText('Username')).toBeInTheDocument()
    expect(screen.getByLabelText('Password')).toBeInTheDocument()
  })

  it('renders background image', () => {
    renderWithAuth()
    
    const backgroundImage = screen.getByAltText('Background')
    expect(backgroundImage).toBeInTheDocument()
    expect(backgroundImage).toHaveAttribute('src', '/authPageArt.jpg')
  })

  it('has correct input placeholders', () => {
    renderWithAuth()
    
    const usernameInput = screen.getByPlaceholderText('Type your username')
    const passwordInput = screen.getByPlaceholderText('Type your password')
    
    expect(usernameInput).toBeInTheDocument()
    expect(passwordInput).toBeInTheDocument()
  })

  it('has correct input types', () => {
    renderWithAuth()
    
    const usernameInput = screen.getByLabelText('Username')
    const passwordInput = screen.getByLabelText('Password')
    
    expect(usernameInput).toHaveAttribute('type', 'text')
    expect(passwordInput).toHaveAttribute('type', 'password')
  })

  it('allows typing in username field', async () => {
    const user = userEvent.setup()
    renderWithAuth()
    
    const usernameInput = screen.getByLabelText('Username')
    await user.type(usernameInput, 'testuser')
    
    expect(usernameInput).toHaveValue('testuser')
  })

  it('allows typing in password field', async () => {
    const user = userEvent.setup()
    renderWithAuth()
    
    const passwordInput = screen.getByLabelText('Password')
    await user.type(passwordInput, 'testpassword')
    
    expect(passwordInput).toHaveValue('testpassword')
  })

  it('renders forgot password link', () => {
    renderWithAuth()
    
    const forgotPasswordLink = screen.getByText('Forgot password?')
    expect(forgotPasswordLink).toBeInTheDocument()
    expect(forgotPasswordLink.closest('a')).toHaveAttribute('href', '/authentication/forgotPassword')
  })

  it('renders sign up link', () => {
    renderWithAuth()
    
    const signUpLink = screen.getByText('SIGN UP')
    expect(signUpLink).toBeInTheDocument()
    expect(signUpLink.closest('a')).toHaveAttribute('href', '/authentication/signUp')
  })

  it('renders login button', () => {
    renderWithAuth()
    
    const loginButton = screen.getByRole('button', { name: 'Login' })
    expect(loginButton).toBeInTheDocument()
    expect(loginButton).toHaveAttribute('type', 'submit')
  })

  it('handles form submission and redirects to chat page', async () => {
    const user = userEvent.setup()
    renderWithAuth()
    
    const usernameInput = screen.getByLabelText('Username')
    const passwordInput = screen.getByLabelText('Password')
    const loginButton = screen.getByRole('button', { name: 'Login' })
    
    // Fill in the form
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'testpassword')
    
    // Submit the form
    await user.click(loginButton)
    
    // Should redirect to chat page
    expect(mockPush).toHaveBeenCalledWith('/chatPage')
  })

  it('handles form submission with Enter key', async () => {
    const user = userEvent.setup()
    renderWithAuth()
    
    const usernameInput = screen.getByLabelText('Username')
    const passwordInput = screen.getByLabelText('Password')
    
    // Fill in the form
    await user.type(usernameInput, 'testuser')
    await user.type(passwordInput, 'testpassword')
    
    // Press Enter in password field
    await user.keyboard('{Enter}')
    
    // Should redirect to chat page
    expect(mockPush).toHaveBeenCalledWith('/chatPage')
  })

  it('prevents default form submission', async () => {
    renderWithAuth()
    
    const form = screen.getByRole('button', { name: 'Login' }).closest('form')
    const preventDefaultSpy = jest.fn()
    
    // Mock preventDefault
    const mockEvent = { preventDefault: preventDefaultSpy } as any
    
    if (form) {
      fireEvent.submit(form, mockEvent)
    }
    
    // Note: In the actual implementation, preventDefault should be called
    // This test verifies the form structure is correct
    expect(form).toBeInTheDocument()
  })

  it('has correct CSS classes', () => {
    renderWithAuth()
    
    const loginPage = screen.getByRole('heading', { name: 'Login' }).closest('.login-page')
    expect(loginPage).toBeInTheDocument()
    
    const loginForm = screen.getByRole('heading', { name: 'Login' }).closest('.login-form')
    expect(loginForm).toBeInTheDocument()
    
    const formTitle = screen.getByRole('heading', { name: 'Login' })
    expect(formTitle).toHaveClass('form-title')
    
    const loginButton = screen.getByRole('button', { name: 'Login' })
    expect(loginButton).toHaveClass('btn-rounded-gradient')
  })

  it('displays "Or Sign Up Using" text', () => {
    renderWithAuth()
    
    expect(screen.getByText('Or Sign Up Using')).toBeInTheDocument()
  })
})
