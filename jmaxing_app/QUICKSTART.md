# J-MAXING Quick Start Guide

Get J-MAXING running in 5 minutes.

---

## Prerequisites

- **Node.js** 18+ ([Download](https://nodejs.org))
- **pnpm** (recommended) or npm/yarn

Install pnpm globally:
```bash
npm install -g pnpm
```

---

## Installation

### 1. Navigate to the app directory
```bash
cd jmaxing_app
```

### 2. Install dependencies
```bash
pnpm install
```

This will install:
- React 18 + TypeScript
- TailwindCSS + Framer Motion
- Monaco Editor
- React Query + Zustand
- And all other dependencies (~500MB node_modules)

**Expected output:**
```
Progress: resolved 523, reused 523, downloaded 0, added 523, done
Done in 15s
```

---

## Running the App

### Start development server
```bash
pnpm dev
```

**Expected output:**
```
  VITE v5.x.x  ready in 1234 ms

  ➜  Local:   http://localhost:3001/
  ➜  Network: use --host to expose
  ➜  press h to show help
```

### Open in browser
Navigate to: **http://localhost:3001**

---

## First Time Setup

### 1. Create environment file (optional)
```bash
# Create .env.local
echo "VITE_API_BASE=http://localhost:8000/api" > .env.local
```

If you skip this, the app will use the default mock API endpoint.

### 2. Login to the app
1. Click **"Get Started"** on homepage
2. Or navigate to `/login`
3. Enter any username (e.g., `testuser`)
4. Click **"Login"**

You'll be redirected to the homepage with 1000 JC starting balance.

---

## Explore the App

### Available Routes

| Route | Description |
|-------|-------------|
| `/` | Homepage with hero and features |
| `/jobs` | Browse available jobs |
| `/jobs/:jobId` | View job details + submit solution |
| `/leaderboard` | Top earners leaderboard |
| `/profile` | Your profile and stats (protected) |
| `/wallet` | JC balance and transactions (protected) |
| `/submit` | Create a new job (protected) |
| `/login` | Authentication |

### Test the Features

#### Browse Jobs
1. Go to `/jobs`
2. See list of code improvement jobs
3. Filter by difficulty or search by keywords
4. Click a job card to view details

#### Submit a Solution
1. Open a job from the job board
2. View the original code (left side)
3. Write improved code (right side, Monaco editor)
4. Click **"Submit Solution"**
5. See your score and payout

#### Check Leaderboard
1. Go to `/leaderboard`
2. See top earners
3. Your rank is highlighted with neon-green border
4. Toggle between Top 10/50/100

#### View Profile
1. Go to `/profile`
2. See your stats, reputation, achievements
3. View submission history
4. Click submissions to revisit jobs

#### Manage Wallet
1. Go to `/wallet`
2. See your JC balance
3. View transaction history
4. Check earning/spending patterns

#### Create a Job
1. Go to `/submit`
2. Paste code that needs improvement
3. Add test cases (JSON format)
4. Set reward and difficulty
5. Click **"Submit Job"**

---

## Development Commands

### Start dev server
```bash
pnpm dev
```

### Build for production
```bash
pnpm build
```

Output will be in `dist/` folder.

### Preview production build
```bash
pnpm preview
```

### Lint code
```bash
pnpm lint
```

### Type check
```bash
pnpm type-check
```

---

## Backend Integration

### Without Backend (Current State)
The app works with **mock data** from the API client. You can:
- Browse jobs (hardcoded sample jobs)
- Login/logout (local auth)
- View leaderboard (mock data)
- View profile (mock submissions)
- View wallet (mock transactions)

### With Janus Service Gateway

#### 1. Start the Service Gateway
```bash
# In another terminal, from Janus root
cd ..
python janus_service_gateway.py
```

This starts the backend on `http://localhost:8000`

#### 2. Update .env.local
```env
VITE_API_BASE=http://localhost:8000/api
```

#### 3. Restart the dev server
```bash
pnpm dev
```

Now the app will connect to the real backend with:
- Real job submissions
- Actual code execution
- Oxpecker quality analysis
- True JC payouts
- Persistent data

---

## Troubleshooting

### Port 3001 already in use
```bash
# Kill the process using port 3001
lsof -ti :3001 | xargs kill -9

# Or use a different port
pnpm dev --port 3002
```

### Monaco Editor not loading
Clear browser cache and reload:
```
Chrome/Edge: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
Firefox: Ctrl+F5 (Windows) or Cmd+Shift+R (Mac)
```

### Styles not applying
```bash
# Rebuild Tailwind CSS
rm -rf node_modules/.vite
pnpm dev
```

### TypeScript errors
```bash
# Regenerate types
pnpm type-check
```

### Dependencies not installing
```bash
# Clear cache and reinstall
rm -rf node_modules pnpm-lock.yaml
pnpm install
```

---

## Mobile Testing

### Test on your phone (same network)

1. Find your local IP:
```bash
# Mac/Linux
ifconfig | grep "inet "

# Windows
ipconfig
```

2. Start dev server with host flag:
```bash
pnpm dev --host
```

3. Open on phone:
```
http://YOUR_IP:3001
```

Example: `http://192.168.1.100:3001`

### PWA Installation

1. Open app in Chrome mobile
2. Tap the menu (3 dots)
3. Select **"Add to Home Screen"**
4. App now launches like a native app

---

## Hot Reload Tips

The dev server supports **Hot Module Replacement (HMR)**:

- **CSS changes** → Instant update (no refresh)
- **React component changes** → Fast refresh (preserves state)
- **Config changes** → Full reload required

If HMR stops working:
1. Save the file again
2. Or refresh the browser
3. Or restart the dev server

---

## File Watching

Vite watches these directories:
- `src/**/*` - All source files
- `public/**/*` - Static assets
- `index.html` - Entry HTML
- `vite.config.ts` - Vite config
- `tailwind.config.js` - Tailwind config

Changes outside these directories won't trigger reloads.

---

## Performance Tips

### Faster Builds
```bash
# Use pnpm (faster than npm/yarn)
pnpm install

# Clear Vite cache if slow
rm -rf node_modules/.vite
```

### Monaco Editor
Monaco Editor is ~1MB. To reduce bundle size:
- Only import languages you need
- Use lightweight alternatives for simple editing
- Or keep Monaco (it's worth it for the UX)

### React Query
Adjust cache times in `src/api/client.ts`:
```typescript
const { data } = useQuery({
  queryKey: ['jobs'],
  queryFn: jobsAPI.list,
  staleTime: 60000,  // Cache for 60s
  cacheTime: 300000, // Keep in memory for 5min
})
```

---

## Next Steps

### For Users
1. Complete a job and earn your first JC
2. Climb the leaderboard
3. Unlock achievements
4. Create your own jobs

### For Developers
1. Read `ARCHITECTURE.md` for technical details
2. Check `COMPLETION_SUMMARY.md` for feature status
3. Review `README.md` for full documentation
4. Explore the codebase in `src/`

### For Contributors
1. Fork the repo
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Useful Links

- **Main README**: `README.md`
- **Architecture**: `ARCHITECTURE.md`
- **Completion Status**: `COMPLETION_SUMMARY.md`
- **Backend Docs**: `../SERVICE_GATEWAY_README.md`
- **Repository Modo**: `../REPOSITORY_MODO.md`

---

## Support

Questions or issues?

- **GitHub**: [@Thoseidiots](https://github.com/Thoseidiots)
- **Email**: legac3y@gmail.com
- **Mesh**: `jmaxing.mesh:3001`

---

**Happy coding! 🚀**

*Remember: Code better. Earn faster. No gatekeepers.*
