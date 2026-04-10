# J-MAXING - Code. Compete. Earn.

**Gamified code improvement marketplace on the Janus mesh network**

Get paid to improve code. Compete with devs worldwide. No resumes. No interviews. Just code and earn Janus Credits.

![J-MAXING Screenshot](./docs/screenshot.png)

---

## 🚀 What is J-MAXING?

J-MAXING is a **Gen-Z focused** code improvement platform where developers earn Janus Credits (JC) by improving code quality. It's built on the Janus mesh network - completely autonomous, no API gatekeepers, no corporate middlemen.

### Why It's Different

✅ **No Applications** - Just submit code, get paid
✅ **Instant Payouts** - JC hits your wallet immediately
✅ **Quality Verified** - Automated testing + static analysis
✅ **Compete Globally** - Leaderboards, reputation, bonuses
✅ **Mesh Native** - Runs on your infrastructure, zero dependencies
✅ **Gen-Z First** - Built for devs who code, not talk

---

## 🎮 Features

### For Code Improvers (Earn JC)
- **Browse Jobs** - Find code that needs improvement
- **Submit Solutions** - Write better code, run tests
- **Get Scored** - Automated evaluation (correctness + quality + improvement)
- **Earn Credits** - JC paid instantly based on score tier
- **Climb Leaderboard** - Top performers get bonus rewards

### For Job Posters (Get Better Code)
- **Post Code** - Upload code that needs improvement
- **Define Tests** - Specify expected behavior
- **Set Reward** - Offer JC for quality improvements
- **Auto-Evaluate** - Oxpecker static analysis + test execution
- **Get Results** - Best solutions delivered automatically

### Scoring System

J-MAXING uses a **three-factor scoring algorithm**:

```
Final Score = 0.4 × Correctness + 0.3 × Quality + 0.3 × Improvement
```

**Correctness** - All tests must pass
**Quality** - Static analysis via Oxpecker (checks for dangerous patterns, complexity, docstrings)
**Improvement** - Delta between original and improved code issues

**Payout Tiers**:
- Score < 0.3 → 0 JC (rejected)
- Score 0.3-0.7 → 50% of reward
- Score 0.7-0.9 → 100% of reward
- Score > 0.9 → 150% of reward (bonus!)

---

## 🛠️ Tech Stack

- **Frontend**: React 18 + TypeScript + Vite
- **Styling**: TailwindCSS + Framer Motion
- **State**: Zustand
- **Data Fetching**: TanStack Query (React Query)
- **Code Editor**: Monaco Editor
- **Backend**: Janus Service Gateway (Python/Flask)
- **Execution**: Nexus Core (Rust/gRPC) + soft_ntb
- **Network**: MeshISP (DHCP, DNS, WiFi hotspot)

---

## 🚀 Quick Start

### Prerequisites
- Node.js 18+
- pnpm (or npm/yarn)
- Janus Service Gateway running on `localhost:8000`

### Installation

```bash
# Clone the repo
git clone https://github.com/Thoseidiots/Janus.git
cd Janus/jmaxing_app

# Install dependencies
pnpm install

# Start dev server
pnpm dev
```

The app will open at `http://localhost:3001`

### Environment Variables

Create `.env.local`:

```env
VITE_API_BASE=http://localhost:8000/api
```

---

## 📁 Project Structure

```
jmaxing_app/
├── src/
│   ├── api/
│   │   └── client.ts          # API client + types
│   ├── components/
│   │   └── Layout.tsx         # Main layout with header/footer
│   ├── pages/
│   │   ├── HomePage.tsx       # Landing page
│   │   ├── JobBoardPage.tsx   # Browse jobs
│   │   ├── JobDetailPage.tsx  # View/submit to job
│   │   ├── LeaderboardPage.tsx# Top earners
│   │   ├── ProfilePage.tsx    # User profile + stats
│   │   ├── WalletPage.tsx     # JC balance + transactions
│   │   ├── SubmitJobPage.tsx  # Post new job
│   │   └── LoginPage.tsx      # Auth
│   ├── store/
│   │   └── authStore.ts       # Zustand auth state
│   ├── App.tsx                # Routes
│   ├── main.tsx               # Entry point
│   └── index.css              # Global styles + utilities
├── public/
│   ├── manifest.json          # PWA manifest
│   └── jmaxing-logo.svg       # Logo
├── package.json
├── vite.config.ts
├── tailwind.config.js
└── README.md
```

---

## 🎨 Design System

### Colors

**Primary Palette**:
- `janus-*` - Blue shades (0ea5e9 → 0c4a6e)
- `neon-green` - #39FF14 (primary accent)
- `neon-pink` - #FF10F0
- `neon-blue` - #00FFFF

**Dark Theme**:
- Background: `gray-950` (#030712)
- Panels: `gray-900` (#111827)
- Text: `gray-100` (#f3f4f6)

### Components

**Glass Panels**:
```tsx
className="glass-panel" // Frosted glass effect
```

**Neon Buttons**:
```tsx
className="btn-neon" // Green neon glow effect
className="btn-primary" // Solid janus-blue
className="btn-secondary" // Gray
```

**Stat Cards**:
```tsx
className="stat-card" // Hover effects + glass
```

**Badges**:
```tsx
className="badge-success" // Green
className="badge-warning" // Yellow
className="badge-error" // Red
className="badge-info" // Blue
```

---

## 🔗 API Integration

The app connects to Janus Service Gateway endpoints:

### Jobs API
```typescript
// List all jobs
GET /api/jobs

// Get job details
GET /api/jobs/:jobId

// Submit solution
POST /api/jobs/:jobId/submit
{
  "output_code": "improved code here"
}

// Create new job
POST /api/jobs
{
  "input_code": "code to improve",
  "tests": [{ "input": [...], "expected": ... }],
  "reward_jc": 100,
  "difficulty": 2.5
}
```

### Wallet API
```typescript
// Get balance
GET /api/credits/balance

// Get transactions
GET /api/credits/transactions
```

### Leaderboard API
```typescript
// Get top earners
GET /api/leaderboard?limit=100
```

---

## 🎯 Roadmap

### Phase 1: MVP (Current)
- [x] Core UI/UX
- [x] Job browsing
- [x] Login/auth
- [x] Wallet display
- [x] Job detail + submission
- [x] Leaderboard
- [x] Profile page
- [x] Submit job page

### Phase 2: Scoring Engine
- [ ] Integrate Oxpecker static analysis
- [ ] Test execution sandbox
- [ ] Real-time scoring
- [ ] Payout automation

### Phase 3: Gamification
- [ ] Achievement system
- [ ] Reputation levels (Rookie → Legend)
- [ ] Daily challenges
- [ ] Streak bonuses
- [ ] Team competitions

### Phase 4: Social
- [ ] User profiles with bio/socials
- [ ] Code reviews + comments
- [ ] Follow top performers
- [ ] Share solutions (after job closes)

### Phase 5: Advanced
- [ ] Multiple language support (Python, JS, Rust, Go)
- [ ] AI-assisted improvement suggestions
- [ ] Live coding competitions
- [ ] Mentorship system

---

## 🌐 Deployment

### Build for Production

```bash
pnpm build
```

Output in `dist/` folder.

### Deploy to MeshISP

```bash
# Build
pnpm build

# Create deployment zip
cd dist && zip -r ../jmaxing.zip . && cd ..

# Upload to Janus Service Gateway
curl -X POST http://gateway.mesh:8000/api/deploy \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "app=@jmaxing.zip"
```

App will be available at `http://jmaxing.mesh` on your mesh network.

### PWA Features

J-MAXING is a **Progressive Web App**:
- ✅ Installable on mobile/desktop
- ✅ Offline support
- ✅ Push notifications (coming soon)
- ✅ Fast, native-like experience

---

## 🤝 Contributing

J-MAXING is open to contributions! Here's how:

1. **Fork the repo**
2. **Create a branch** (`git checkout -b feature/amazing-feature`)
3. **Commit changes** (`git commit -m 'Add amazing feature'`)
4. **Push** (`git push origin feature/amazing-feature`)
5. **Open a Pull Request**

### Development Guidelines

- Use TypeScript strict mode
- Follow existing code style (Prettier + ESLint)
- Write descriptive commit messages
- Test on both desktop and mobile
- Keep components under 300 lines

---

## 🐛 Known Issues

- [ ] Monaco Editor performance on mobile
- [ ] Leaderboard pagination needed for > 1000 users
- [ ] Profile page stats not real-time
- [ ] Wallet transaction history limited to 100

---

## 📄 License

MIT License - see [LICENSE](./LICENSE)

---

## 🙏 Acknowledgments

Built with:
- [React](https://react.dev)
- [TailwindCSS](https://tailwindcss.com)
- [Framer Motion](https://www.framer.com/motion)
- [Monaco Editor](https://microsoft.github.io/monaco-editor)
- [Janus Mesh Network](https://github.com/Thoseidiots/Janus)

---

## 📧 Contact

Questions? Ideas? Want to contribute?

- GitHub: [@Thoseidiots](https://github.com/Thoseidiots)
- Email: legac3y@gmail.com
- Mesh: `jmaxing.mesh:3001`

---

**J-MAXING** - Code better. Earn faster. No gatekeepers.

Powered by [Janus](https://github.com/Thoseidiots/Janus) • Built for Gen-Z devs • 100% autonomous
