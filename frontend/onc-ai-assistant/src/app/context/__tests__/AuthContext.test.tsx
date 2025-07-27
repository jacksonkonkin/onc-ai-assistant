import { render, screen, act } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { AuthProvider, useAuth } from '../AuthContext'

// Test component that uses the AuthContext
const TestComponent = () => {
  const { isLoggedIn, setIsLoggedIn } = useAuth()
  
  return (
    <div>
      <div data-testid="login-status">
        {isLoggedIn ? 'Logged In' : 'Logged Out'}
      </div>
      <button onClick={() => setIsLoggedIn(true)} data-testid="login-btn">
        Login
      </button>
      <button onClick={() => setIsLoggedIn(false)} data-testid="logout-btn">
        Logout
      </button>
    </div>
  )
}

describe('AuthContext', () => {
  it('provides initial logged out state', () => {
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    )
    
    expect(screen.getByTestId('login-status')).toHaveTextContent('Logged Out')
  })

  it('allows login state to be changed to true', async () => {
    const user = userEvent.setup()
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    )
    
    const loginButton = screen.getByTestId('login-btn')
    
    await user.click(loginButton)
    
    expect(screen.getByTestId('login-status')).toHaveTextContent('Logged In')
  })

  it('allows login state to be changed to false', async () => {
    const user = userEvent.setup()
    render(
      <AuthProvider>
        <TestComponent />
      </AuthProvider>
    )
    
    const loginButton = screen.getByTestId('login-btn')
    const logoutButton = screen.getByTestId('logout-btn')
    
    // First login
    await user.click(loginButton)
    expect(screen.getByTestId('login-status')).toHaveTextContent('Logged In')
    
    // Then logout
    await user.click(logoutButton)
    expect(screen.getByTestId('login-status')).toHaveTextContent('Logged Out')
  })

  it('throws error when useAuth is used outside AuthProvider', () => {
    // Mock console.error to avoid polluting test output
    const consoleSpy = jest.spyOn(console, 'error').mockImplementation(() => {})
    
    const ComponentWithoutProvider = () => {
      try {
        useAuth()
        return <div>Should not render</div>
      } catch (error) {
        return <div data-testid="error-message">{(error as Error).message}</div>
      }
    }
    
    render(<ComponentWithoutProvider />)
    
    expect(screen.getByTestId('error-message')).toHaveTextContent(
      'useAuth must be used within AuthProvider'
    )
    
    consoleSpy.mockRestore()
  })

  it('maintains state across multiple components', async () => {
    const user = userEvent.setup()
    
    const FirstComponent = () => {
      const { isLoggedIn, setIsLoggedIn } = useAuth()
      return (
        <div>
          <div data-testid="first-status">{isLoggedIn ? 'First: Logged In' : 'First: Logged Out'}</div>
          <button onClick={() => setIsLoggedIn(true)} data-testid="first-login">Login from First</button>
        </div>
      )
    }
    
    const SecondComponent = () => {
      const { isLoggedIn } = useAuth()
      return (
        <div data-testid="second-status">{isLoggedIn ? 'Second: Logged In' : 'Second: Logged Out'}</div>
      )
    }
    
    render(
      <AuthProvider>
        <FirstComponent />
        <SecondComponent />
      </AuthProvider>
    )
    
    // Both components should show logged out initially
    expect(screen.getByTestId('first-status')).toHaveTextContent('First: Logged Out')
    expect(screen.getByTestId('second-status')).toHaveTextContent('Second: Logged Out')
    
    // Login from first component
    await user.click(screen.getByTestId('first-login'))
    
    // Both components should show logged in
    expect(screen.getByTestId('first-status')).toHaveTextContent('First: Logged In')
    expect(screen.getByTestId('second-status')).toHaveTextContent('Second: Logged In')
  })
})
