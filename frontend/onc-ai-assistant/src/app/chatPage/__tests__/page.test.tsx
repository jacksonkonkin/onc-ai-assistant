import { render, screen, fireEvent, waitFor } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import ChatPage from '../page'

// Mock the fetch function
global.fetch = jest.fn()

describe('ChatPage', () => {
  beforeEach(() => {
    // Reset the mock before each test
    jest.clearAllMocks()
  })

  it('renders the chat page with initial elements', () => {
    render(<ChatPage />)
    
    // Check if the greeting message is present
    expect(screen.getByText('Hello! How can I assist you today?')).toBeInTheDocument()
    
    // Check if the input field is present
    expect(screen.getByPlaceholderText('Type a message...')).toBeInTheDocument()
    
    // Check if the send button is present
    expect(screen.getByRole('button')).toBeInTheDocument()
  })

  it('allows user to type in the input field', async () => {
    const user = userEvent.setup()
    render(<ChatPage />)
    
    const input = screen.getByPlaceholderText('Type a message...')
    await user.type(input, 'Hello, AI!')
    
    expect(input).toHaveValue('Hello, AI!')
  })

  it('handles API errors gracefully', async () => {
    const user = userEvent.setup()
    
    // Mock API error
    ;(global.fetch as jest.Mock).mockRejectedValueOnce(new Error('API Error'))
    
    render(<ChatPage />)
    
    const input = screen.getByPlaceholderText('Type a message...')
    const sendButton = screen.getByRole('button')
    
    await user.type(input, 'Test message')
    await user.click(sendButton)
    
    // Wait for error message to appear
    await waitFor(() => {
      expect(screen.getByText('Sorry, something went wrong.')).toBeInTheDocument()
    })
  })

  it('does not send empty messages', async () => {
    const user = userEvent.setup()
    render(<ChatPage />)
    
    const sendButton = screen.getByRole('button')
    
    // Try to send empty message
    await user.click(sendButton)
    
    // Should not make API call
    expect(global.fetch).not.toHaveBeenCalled()
  })
})
