# J-MAXING Architecture

Complete technical documentation for the J-MAXING React PWA frontend.

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     J-MAXING Frontend                        │
│                   (React 18 + TypeScript)                    │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ HTTP/REST
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              Janus Service Gateway (Port 8000)               │
│                    (Python/Flask)                            │
└─────────────────────────────────────────────────────────────┘
                            │
                            │ gRPC
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                     Nexus Core                               │
│              (Rust + Raft + WASM + LAS)                      │
└─────────────────────────────────────────────────────────────┘
                            │
                            │
        ┌───────────────────┼───────────────────┐
        │                   │                   │
        ▼                   ▼                   ▼
  ┌──────────┐       ┌──────────┐       ┌──────────┐
  │ Oxpecker │       │soft_ntb  │       │  MeshISP │
  │ (Static  │       │(TCP/NTB) │       │  (DNS/   │
  │ Analysis)│       │          │       │  DHCP)   │
  └──────────┘       └──────────┘       └──────────┘
```

---

## 📁 Project Structure

```
jmaxing_app/
├── src/
│   ├── api/
│   │   └── client.ts              # Axios client + API functions
│   ├── components/
│   │   └── Layout.tsx             # Main layout wrapper
│   ├── pages/
│   │   ├── HomePage.tsx           # Landing page (348 lines)
│   │   ├── JobBoardPage.tsx       # Job listings (285 lines)
│   │   ├── JobDetailPage.tsx      # Job detail + submission (348 lines)
│   │   ├── LeaderboardPage.tsx    # Rankings (303 lines)
│   │   ├── ProfilePage.tsx        # User profile (287 lines)
│   │   ├── WalletPage.tsx         # Wallet + transactions (263 lines)
│   │   ├── SubmitJobPage.tsx      # Create job (368 lines)
│   │   └── LoginPage.tsx          # Authentication (150 lines)
│   ├── store/
│   │   └── authStore.ts           # Zustand auth state
│   ├── App.tsx                    # Router configuration
│   ├── main.tsx                   # React entry point
│   └── index.css                  # Global styles + theme
├── public/
│   ├── manifest.json              # PWA manifest
│   └── jmaxing-logo.svg           # Logo asset
├── package.json                   # Dependencies
├── vite.config.ts                 # Vite configuration
├── tailwind.config.js             # TailwindCSS config
├── tsconfig.json                  # TypeScript config
├── README.md                      # Documentation
├── COMPLETION_SUMMARY.md          # Feature completion status
└── ARCHITECTURE.md                # This file
```

---

## 🔌 API Client Architecture

### Axios Instance
**File**: `src/api/client.ts`

```typescript
const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_BASE || 'http://localhost:8000/api',
})

// Auto-inject JWT from localStorage
apiClient.interceptors.request.use((config) => {
  const token = getTokenFromStorage()
  if (token) {
    config.headers.Authorization = `Bearer ${token}`
  }
  return config
})
```

### API Modules

#### 1. Jobs API
```typescript
jobsAPI.list()                    // GET /api/jobs
jobsAPI.get(jobId)                // GET /api/jobs/:jobId
jobsAPI.submit(jobId, code)       // POST /api/jobs/:jobId/submit
jobsAPI.create(job)               // POST /api/jobs
```

#### 2. Submissions API
```typescript
submissionsAPI.getForJob(jobId)   // GET /api/jobs/:jobId/submissions
submissionsAPI.getMySubmissions() // GET /api/submissions/me
```

#### 3. Leaderboard API
```typescript
leaderboardAPI.get(limit)         // GET /api/leaderboard?limit=100
```

#### 4. Wallet API
```typescript
walletAPI.getBalance()            // GET /api/credits/balance
walletAPI.getTransactions()       // GET /api/credits/transactions
```

---

## 🎯 State Management

### Zustand Auth Store
**File**: `src/store/authStore.ts`

```typescript
interface AuthState {
  user: User | null
  token: string | null
  login: (username: string) => void
  logout: () => void
  updateBalance: (balance: number) => void
  updateUser: (updates: Partial<User>) => void
}

// Persisted to localStorage as 'jmaxing-auth'
export const useAuthStore = create<AuthState>()(
  persist(/* ... */, { name: 'jmaxing-auth' })
)
```

### React Query Integration

All pages use TanStack Query for server state:

```typescript
const { data, isLoading } = useQuery({
  queryKey: ['jobs'],
  queryFn: jobsAPI.list,
  staleTime: 30000, // 30s cache
})

const mutation = useMutation({
  mutationFn: jobsAPI.submit,
  onSuccess: (data) => {
    queryClient.invalidateQueries(['jobs'])
  },
})
```

---

## 🎨 Design System

### Color Palette

```css
/* Primary Neon Colors */
--neon-green: #39FF14;
--neon-pink: #FF10F0;
--neon-blue: #00FFFF;

/* Janus Blue Scale */
--janus-50: #f0f9ff;
--janus-500: #0ea5e9;
--janus-900: #0c4a6e;

/* Dark Theme */
--gray-950: #030712;  /* Background */
--gray-900: #111827;  /* Panels */
--gray-800: #1f2937;  /* Borders */
```

### Component Classes

#### Glass Panels
```css
.glass-panel {
  @apply bg-gray-900/50 backdrop-blur-sm border border-gray-800
         rounded-xl shadow-xl;
}
```

#### Neon Buttons
```css
.btn-neon {
  @apply bg-neon-green text-gray-950 px-6 py-3 rounded-lg
         font-semibold hover:bg-neon-green/90
         shadow-[0_0_20px_rgba(57,255,20,0.3)]
         transition-all;
}
```

#### Badges
```css
.badge-success { @apply bg-green-500/10 text-green-400 ... }
.badge-warning { @apply bg-yellow-500/10 text-yellow-400 ... }
.badge-error   { @apply bg-red-500/10 text-red-400 ... }
.badge-info    { @apply bg-blue-500/10 text-blue-400 ... }
```

#### Stat Cards
```css
.stat-card {
  @apply glass-panel p-6 hover:shadow-2xl
         hover:border-neon-green/30
         transition-all duration-300;
}
```

---

## 🔐 Authentication Flow

```
┌─────────┐
│ User    │
└────┬────┘
     │
     │ 1. Visit /profile (protected route)
     ▼
┌─────────────────┐
│ ProtectedRoute  │
│ Checks user     │──────── No user ────▶ Navigate to /login
│ in authStore    │
└────┬────────────┘
     │
     │ User exists
     ▼
┌─────────────────┐
│ Render Profile  │
│ Page            │
└─────────────────┘
```

### Login Flow
```typescript
// LoginPage.tsx
const handleLogin = (username: string) => {
  const mockUser = {
    id: `user_${username}`,
    username,
    balance: 1000,
    // ... other fields
  }

  login(username)  // Zustand action
  navigate('/')    // Redirect to home
}
```

### Protected Routes
```typescript
// App.tsx
<Route path="profile" element={
  <ProtectedRoute>
    <ProfilePage />
  </ProtectedRoute>
} />
```

---

## 🧩 Component Architecture

### Layout Wrapper
**File**: `src/components/Layout.tsx`

```
┌──────────────────────────────────────────────┐
│              Header                           │
│  [Logo] [Nav] [Balance] [Profile Dropdown]   │
├──────────────────────────────────────────────┤
│                                               │
│              <Outlet />                       │
│          (Child Route Renders Here)           │
│                                               │
├──────────────────────────────────────────────┤
│              Footer                           │
│    [Links] [Socials] [Copyright]             │
└──────────────────────────────────────────────┘
```

### Page Component Pattern

All pages follow this structure:

```typescript
export default function PageName() {
  // 1. Hooks
  const { user } = useAuthStore()
  const navigate = useNavigate()

  // 2. Data fetching (React Query)
  const { data, isLoading } = useQuery({ ... })

  // 3. Mutations
  const mutation = useMutation({ ... })

  // 4. Event handlers
  const handleAction = () => { ... }

  // 5. Loading/empty states
  if (isLoading) return <LoadingSpinner />
  if (!data) return <EmptyState />

  // 6. Main render with Framer Motion
  return (
    <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }}>
      {/* Content */}
    </motion.div>
  )
}
```

---

## 📊 Data Flow

### Job Submission Flow

```
User writes code in JobDetailPage
         │
         ▼
Click "Submit Solution"
         │
         ▼
jobsAPI.submit(jobId, code)
         │
         ▼
POST /api/jobs/:jobId/submit
  { output_code: "..." }
         │
         ▼
Janus Service Gateway
         │
         ├─▶ Execute tests (Nexus Core)
         ├─▶ Run Oxpecker analysis
         └─▶ Calculate score
         │
         ▼
Return { score, payout, status }
         │
         ▼
Update UI with result
Navigate to /jobs
```

### Balance Updates

```
Submission accepted
         │
         ▼
Service Gateway credits user
         │
         ▼
Frontend polls /api/credits/balance
         │
         ▼
authStore.updateBalance(newBalance)
         │
         ▼
UI reactively updates:
  - Header balance display
  - Wallet page
  - Profile stats
```

---

## 🚀 Build & Deployment

### Development
```bash
pnpm dev             # Start dev server (port 3001)
pnpm build           # Build for production
pnpm preview         # Preview production build
```

### Production Build
```bash
pnpm build

# Output: dist/ folder
dist/
├── assets/
│   ├── index-[hash].js      # ~500KB (React + Monaco)
│   ├── index-[hash].css     # ~50KB (TailwindCSS)
│   └── vendor-[hash].js     # ~1.2MB (Dependencies)
├── index.html
└── manifest.json
```

### Deploy to MeshISP

```bash
# 1. Build
pnpm build

# 2. Create deployment zip
cd dist && zip -r ../jmaxing.zip . && cd ..

# 3. Upload to Janus Service Gateway
curl -X POST http://gateway.mesh:8000/api/deploy \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "app=@jmaxing.zip"

# 4. Access at http://jmaxing.mesh:3001
```

---

## 🔧 Environment Variables

**File**: `.env.local`

```env
# API endpoint for Janus Service Gateway
VITE_API_BASE=http://localhost:8000/api

# Production
# VITE_API_BASE=http://gateway.mesh:8000/api
```

---

## 📱 PWA Configuration

**File**: `public/manifest.json`

```json
{
  "name": "J-MAXING",
  "short_name": "J-MAX",
  "description": "Code. Compete. Earn on Janus mesh",
  "start_url": "/",
  "display": "standalone",
  "background_color": "#030712",
  "theme_color": "#39FF14",
  "icons": [ ... ]
}
```

### Service Worker (Vite PWA Plugin)
```typescript
// vite.config.ts
import { VitePWA } from 'vite-plugin-pwa'

export default defineConfig({
  plugins: [
    react(),
    VitePWA({
      registerType: 'autoUpdate',
      workbox: {
        globPatterns: ['**/*.{js,css,html,ico,png,svg}'],
      },
    }),
  ],
})
```

---

## 🧪 Testing Strategy

### Unit Tests (Future)
```bash
vitest                # Run unit tests
vitest --coverage     # Generate coverage
```

### E2E Tests (Future)
```bash
playwright test       # Run E2E tests
```

### Manual Testing Checklist

- [ ] Login/logout flow
- [ ] Job browsing and filtering
- [ ] Job submission with validation
- [ ] Leaderboard pagination
- [ ] Profile stats accuracy
- [ ] Wallet transaction history
- [ ] Submit job form validation
- [ ] Protected route redirects
- [ ] Mobile responsiveness
- [ ] Dark theme consistency

---

## 🔒 Security Considerations

### Current Implementation
- JWT tokens stored in localStorage (zustand persist)
- Token auto-injection via Axios interceptor
- Protected routes with auth guards
- No sensitive data in URL parameters

### Production Recommendations
1. **HTTPS Only** - Enforce TLS for all connections
2. **CSP Headers** - Implement Content Security Policy
3. **Rate Limiting** - Throttle API requests client-side
4. **Input Sanitization** - Validate all user inputs (especially code)
5. **XSS Prevention** - Already using React (auto-escapes)
6. **CORS** - Configure proper CORS on backend

---

## 📈 Performance Optimizations

### Current Optimizations
- **Code Splitting** - Automatic via Vite
- **Tree Shaking** - Dead code elimination
- **Lazy Loading** - Monaco Editor loaded on demand
- **Image Optimization** - SVG logo (scalable, small)
- **Font Optimization** - System fonts (no web fonts)

### Future Optimizations
- [ ] Implement virtual scrolling for large leaderboards
- [ ] Add image lazy loading for user avatars
- [ ] Compress Monaco Editor bundle (currently ~1MB)
- [ ] Add service worker caching for offline support
- [ ] Implement route-based code splitting

---

## 🐛 Known Issues & Limitations

### Current Limitations
1. **Monaco Editor Mobile** - Performance degradation on low-end devices
2. **Mock Data** - API responses are currently mocked (needs backend)
3. **Leaderboard Pagination** - No server-side pagination (client-side only)
4. **Wallet History** - Limited to 100 transactions
5. **No Real-Time Updates** - No WebSocket for live score updates

### Browser Support
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+
- ⚠️ IE11 not supported

---

## 📚 Dependencies

### Core
- `react` - UI library
- `react-dom` - React rendering
- `react-router-dom` - Routing
- `typescript` - Type safety

### State Management
- `zustand` - Lightweight state
- `@tanstack/react-query` - Server state

### UI/Styling
- `tailwindcss` - Utility-first CSS
- `framer-motion` - Animations
- `@monaco-editor/react` - Code editor

### HTTP
- `axios` - HTTP client

### Build Tools
- `vite` - Build tool
- `@vitejs/plugin-react` - React plugin
- `vite-plugin-pwa` - PWA support

---

## 🎓 Learning Resources

### For Contributors
- [React Docs](https://react.dev)
- [TailwindCSS Docs](https://tailwindcss.com)
- [Framer Motion Docs](https://www.framer.com/motion)
- [React Query Docs](https://tanstack.com/query)
- [Zustand Docs](https://zustand-demo.pmnd.rs)

### Janus-Specific
- `REPOSITORY_MODO.md` - Janus development philosophy
- `janus_service_gateway.py` - Backend API reference
- `SERVICE_GATEWAY_README.md` - Gateway documentation

---

## 📞 Support

For technical questions:
- GitHub: [@Thoseidiots](https://github.com/Thoseidiots)
- Email: legac3y@gmail.com
- Mesh: `jmaxing.mesh:3001`

---

**J-MAXING Architecture v1.0**
*Built for the Janus mesh network • 100% autonomous • No gatekeepers*
