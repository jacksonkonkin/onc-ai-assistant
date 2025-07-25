# Testing Setup

This project is configured with Jest and React Testing Library for comprehensive testing of React components and utilities.

## Installed Testing Dependencies

- **Jest**: JavaScript testing framework
- **@testing-library/react**: Testing utilities for React components
- **@testing-library/jest-dom**: Custom Jest matchers for DOM elements
- **@testing-library/user-event**: Advanced user interaction simulation
- **jest-environment-jsdom**: JSDOM environment for Jest
- **@types/jest**: TypeScript definitions for Jest

## Test Scripts

- `npm test` - Run all tests once
- `npm run test:watch` - Run tests in watch mode (reruns when files change)
- `npm run test:coverage` - Run tests with coverage report

## Configuration Files

- `jest.config.js` - Jest configuration with Next.js integration
- `jest.setup.js` - Global test setup (imports jest-dom matchers)

## Test File Structure

Tests should be placed in `__tests__` directories or named with `.test.ts/.test.tsx` extensions:

```
src/
├── app/
│   └── chatPage/
│       ├── __tests__/
│       │   └── page.test.tsx
│       └── page.tsx
└── utils/
    ├── __tests__/
    │   └── helpers.test.ts
    └── helpers.ts
```

## Writing Tests

### Component Testing Example

```tsx
import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import MyComponent from '../MyComponent'

describe('MyComponent', () => {
  it('renders correctly', () => {
    render(<MyComponent />)
    expect(screen.getByText('Hello World')).toBeInTheDocument()
  })

  it('handles user interactions', async () => {
    const user = userEvent.setup()
    render(<MyComponent />)
    
    await user.click(screen.getByRole('button'))
    expect(screen.getByText('Button clicked')).toBeInTheDocument()
  })
})
```

### Utility Function Testing Example

```ts
import { myUtilFunction } from '../utils'

describe('myUtilFunction', () => {
  it('returns expected result', () => {
    expect(myUtilFunction('input')).toBe('expected output')
  })
})
```

## Best Practices

1. **Test Behavior, Not Implementation**: Focus on testing what the component does, not how it does it
2. **Use User-Centric Queries**: Prefer `getByRole`, `getByLabelText`, etc. over `getByTestId`
3. **Mock External Dependencies**: Use Jest mocks for API calls, external libraries, etc.
4. **Test Edge Cases**: Include tests for error states, empty states, and boundary conditions
5. **Keep Tests Simple**: Each test should focus on one specific behavior

## Mocking API Calls

```tsx
// Mock fetch globally
global.fetch = jest.fn()

// In your test
;(global.fetch as jest.Mock).mockResolvedValueOnce({
  ok: true,
  json: async () => ({ data: 'mocked response' })
})
```

## Coverage Reports

Run `npm run test:coverage` to generate a coverage report. The report will show:
- Statement coverage
- Branch coverage
- Function coverage
- Line coverage

Coverage reports are generated in the `coverage/` directory.
