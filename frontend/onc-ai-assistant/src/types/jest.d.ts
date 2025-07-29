// Type declarations for Jest DOM matchers
import '@testing-library/jest-dom'

declare global {
  namespace jest {
    interface Matchers<R> {
      toBeInTheDocument(): R
      toHaveClass(className: string): R
      toHaveAttribute(attribute: string, value?: any): R
      toHaveTextContent(text: string | RegExp): R
      toHaveValue(value: any): R
    }
  }
}

export {}
