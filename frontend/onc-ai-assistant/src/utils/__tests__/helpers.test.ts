import { formatMessage, validateInput, isValidUrl } from '../helpers'

describe('Helper Functions', () => {
  describe('formatMessage', () => {
    it('capitalizes the first letter of a message', () => {
      expect(formatMessage('hello world')).toBe('Hello world')
    })

    it('trims whitespace and capitalizes', () => {
      expect(formatMessage('  hello world  ')).toBe('Hello world')
    })

    it('handles empty strings', () => {
      expect(formatMessage('')).toBe('')
    })

    it('handles single character', () => {
      expect(formatMessage('a')).toBe('A')
    })
  })

  describe('validateInput', () => {
    it('returns true for valid input', () => {
      expect(validateInput('Hello, how are you?')).toBe(true)
    })

    it('returns false for empty string', () => {
      expect(validateInput('')).toBe(false)
    })

    it('returns false for whitespace only', () => {
      expect(validateInput('   ')).toBe(false)
    })

    it('returns false for input longer than 1000 characters', () => {
      const longInput = 'a'.repeat(1001)
      expect(validateInput(longInput)).toBe(false)
    })

    it('returns true for input exactly 1000 characters', () => {
      const maxInput = 'a'.repeat(1000)
      expect(validateInput(maxInput)).toBe(true)
    })
  })

  describe('isValidUrl', () => {
    it('returns true for valid HTTP URLs', () => {
      expect(isValidUrl('https://example.com')).toBe(true)
      expect(isValidUrl('http://example.com')).toBe(true)
    })

    it('returns true for valid URLs with paths', () => {
      expect(isValidUrl('https://example.com/path/to/resource')).toBe(true)
    })

  })
})
