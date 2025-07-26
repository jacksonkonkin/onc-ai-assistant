import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import AccountDropdown from '../accountDropdown'
import { AuthProvider } from '../../context/AuthContext'

// Create a test wrapper with AuthProvider
const TestWrapper = ({ children }: { children: React.ReactNode }) => (
  <AuthProvider>
    {children}
  </AuthProvider>
)

describe('AccountDropdown', () => {
  const renderWithAuth = () => {
    return render(
      <TestWrapper>
        <AccountDropdown />
      </TestWrapper>
    )
  }

  beforeEach(() => {
    jest.clearAllMocks()
  })

  it('displays account settings link', () => {
    renderWithAuth()
    
    const accountSettingsLink = screen.getByText('Account Settings')
    expect(accountSettingsLink).toBeInTheDocument()
    expect(accountSettingsLink.closest('a')).toHaveAttribute('href', '/accountSettings')
  })

  it('displays logout button', () => {
    renderWithAuth()
    
    const logoutButton = screen.getByText('Log out')
    expect(logoutButton).toBeInTheDocument()
    expect(logoutButton.tagName).toBe('BUTTON')
  })

  it('has correct CSS classes', () => {
    renderWithAuth()
    
    const container = screen.getByText('Account Settings').closest('.account-dropdown')
    expect(container).toHaveClass('account-dropdown')
    
    const dropdown = screen.getByText('Account Settings').closest('.dropdown-menu')
    expect(dropdown).toHaveClass('dropdown-menu')
  })
})
