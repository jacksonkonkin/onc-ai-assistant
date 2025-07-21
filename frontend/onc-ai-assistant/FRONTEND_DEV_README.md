# Frontend Development Without Backend

This setup allows you to develop and test the frontend authentication system without running the backend server.

## Quick Start

### Option 1: Enable Mock Auth (Recommended)
1. Set `useMockAuth = true` in `/src/app/config/devConfig.ts` (already set by default)
2. Start the frontend: `npm run dev`
3. Navigate to `/authentication`
4. Use any of the test usernames with any password

### Option 2: Use Environment Variables
1. Copy `.env.example` to `.env.local`
2. Set `NEXT_PUBLIC_USE_MOCK_AUTH=true`
3. Start the frontend: `npm run dev`

## Test Credentials

When mock auth is enabled, you can log in with these usernames (any password works):

| Username | Role | Description |
|----------|------|-------------|
| `admin` | admin | Full admin access, can see admin panel |
| `generaluser` | general | Regular user access |
| `studentuser` | student | Student role |
| `researcheruser` | researcher | Researcher role |
| `educatoruser` | educator | Educator role |
| `policyuser` | policy-maker | Policy maker role |

## Features Available in Mock Mode

✅ **Login/Logout** - Full authentication flow  
✅ **Role-based access** - Different user roles work correctly  
✅ **Admin panel access** - Admin users can access admin pages  
✅ **JWT tokens** - Mock JWT tokens for consistent behavior  
✅ **Signup flow** - Multi-step signup process works  
✅ **Persistent sessions** - Login state persists across page refreshes  

## Switching Back to Real Backend

When you want to use the real backend again:

1. **Option A:** Change in code
   - Set `useMockAuth = false` in `/src/app/config/devConfig.ts`

2. **Option B:** Use environment variable
   - Set `NEXT_PUBLIC_USE_MOCK_AUTH=false` in `.env.local`

3. **Start backend server**
   - Make sure your backend is running on `http://localhost:8000`

## File Structure

```
src/app/
├── config/
│   └── devConfig.ts          # Toggle mock/real auth
├── services/
│   ├── authService.ts        # Main auth service (auto-switches)
│   └── mockAuthService.ts    # Mock authentication
└── components/
    └── DevCredentialsHelper.tsx  # Development helper (shows test creds)
```

## Mock Service Details

The mock service (`mockAuthService.ts`) provides:
- Fake JWT tokens (not cryptographically secure, for development only)
- Different user roles based on username
- Simulated API delays for realistic testing
- Compatible interface with real backend

## Development Helper

When mock auth is enabled, you'll see a blue helper box in the top-right corner of the login page showing available test usernames. This only appears in development mode.

## Troubleshooting

**Problem:** Login fails even with test credentials  
**Solution:** Check that `useMockAuth` is set to `true` in `devConfig.ts`

**Problem:** Can't access admin panel with admin user  
**Solution:** Make sure you're using the username `admin` exactly

**Problem:** Changes not taking effect  
**Solution:** Restart the development server (`npm run dev`)
