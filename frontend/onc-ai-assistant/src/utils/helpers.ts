// Example utility functions for testing
export const formatMessage = (text: string): string => {
  return text.trim().charAt(0).toUpperCase() + text.trim().slice(1)
}

export const validateInput = (input: string): boolean => {
  return input.trim().length > 0 && input.trim().length <= 1000
}

export const isValidUrl = (url: string): boolean => {
  try {
    new URL(url)
    return true
  } catch {
    return false
  }
}
