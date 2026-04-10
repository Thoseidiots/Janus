# J-MAXING Social Features & Referral System

Complete documentation for the new social media and referral features added to J-MAXING.

---

## 🌟 Overview

J-MAXING is now a **social coding platform** where developers can:
- Share code and achievements
- Follow other developers
- Earn JC through referrals
- Build reputation through community engagement
- Showcase their best work

---

## 📱 New Features

### 1. **Social Feed** (`/feed`)
**File**: `src/pages/FeedPage.tsx`

A Twitter-like feed showing activity from developers you follow:

#### Features:
- **Two Tabs**:
  - **Following**: Posts from developers you follow
  - **Explore**: Discover posts from the entire community

- **Create Posts**:
  - Share text updates
  - Attach code snippets with Monaco editor
  - Post about achievements and earnings

- **Interactions**:
  - ❤️ Like posts
  - 💬 Comment on posts
  - 🔄 Share posts
  - View job submissions directly from posts

- **Sidebar**:
  - Your stats (balance, reputation, rank)
  - Trending hashtags
  - Suggested users to follow

#### Example Post Types:
```typescript
// Status Update
{
  type: 'status',
  content: 'Just optimized this sorting algorithm!',
  code: '...',  // Optional code snippet
}

// Job Submission Showcase
{
  type: 'submission',
  content: 'Crushed this refactoring challenge 🔥',
  code: '...',
  score: 0.95,
  payout: 150,
  jobId: '...',
}
```

---

### 2. **Referral System** (`/referrals`)
**File**: `src/pages/ReferralPage.tsx`

Earn Janus Credits by inviting friends to the platform.

#### How It Works:

**Step 1: Share Your Link**
- Get a unique referral code (e.g., `JMAX-ABC123`)
- Share your personalized referral link
- Use built-in social share buttons (Twitter, Facebook, LinkedIn)

**Step 2: Friend Joins**
- They sign up using your link
- They get **50 JC bonus** to start

**Step 3: You Both Earn**
- When they complete their first job, you both earn **100 JC**
- Instant payout, no waiting

#### Rewards Structure:

| Referrals | Bonus Reward | Badge | Special Perk |
|-----------|--------------|-------|--------------|
| 5 | +500 JC | "Influencer" | - |
| 10 | +1,200 JC | "Community Builder" | - |
| 25 | +3,500 JC | "Ambassador" | +10% earnings boost |

#### Referral Stats Dashboard:
- Total referrals (completed + pending)
- Total earned from referrals
- Referral history with timestamps
- Pending referrals status

#### API Endpoints:
```typescript
// Get referral stats
GET /api/referrals/stats
Response: {
  totalReferrals: number
  completedReferrals: number
  pendingReferrals: number
  totalEarned: number
  referralCode: string
  referralLink: string
}

// Get referral history
GET /api/referrals
Response: Referral[]

// Generate new referral code
POST /api/referrals/generate

// Apply referral code (during signup)
POST /api/referrals/apply
Body: { code: string }
```

---

### 3. **User Profiles** (`/users/:userId`)
**File**: `src/pages/UserProfilePage.tsx`

Public profile pages for every developer on J-MAXING.

#### Profile Components:

**Header Section**:
- Avatar (gradient with initials)
- Username and bio
- Location, website, GitHub links
- Follower/following counts
- Follow/unfollow button

**Stats Grid**:
- Global rank
- Reputation score
- Tasks completed
- Current JC balance

**Tabs**:
1. **Posts**: User's activity feed
2. **Followers**: List of followers
3. **Following**: List of users they follow

#### Features:
- View anyone's public profile
- Follow/unfollow from profile page
- See their code submissions and achievements
- Click through to view specific jobs they completed

---

### 4. **Follow System**

#### API Functions:
```typescript
// Follow a user
POST /api/users/:userId/follow

// Unfollow a user
DELETE /api/users/:userId/follow

// Get followers
GET /api/users/:userId/followers
Response: UserProfile[]

// Get following
GET /api/users/:userId/following
Response: UserProfile[]
```

#### Features:
- Follow developers to see their activity in your feed
- Followers/following counts displayed on profiles
- Suggested users to follow in feed sidebar
- Follow button on user cards and profile pages

---

### 5. **Social Interactions**

#### Likes:
```typescript
// Like a post
POST /api/posts/:postId/like

// Unlike a post
DELETE /api/posts/:postId/like
```

- Heart icon (🤍 → ❤️)
- Like counter
- Liked state persisted

#### Comments:
```typescript
// Get comments
GET /api/posts/:postId/comments
Response: Comment[]

// Add comment
POST /api/posts/:postId/comments
Body: { content: string }

// Delete comment
DELETE /api/posts/:postId/comments/:commentId

// Like comment
POST /api/posts/:postId/comments/:commentId/like
```

- Nested comments support
- Like comments
- Reply to comments
- Real-time comment counts

#### Shares:
```typescript
// Share a post
POST /api/posts/:postId/share
```

- Share counter
- Reshare to your followers
- Share to external platforms

---

## 🔧 Technical Implementation

### API Client
**File**: `src/api/social.ts`

All social features use a centralized API client:

```typescript
import { socialAPI } from '../api/social'

// User operations
const profile = await socialAPI.getProfile(userId)
await socialAPI.updateProfile({ bio: '...' })

// Follow operations
await socialAPI.follow(userId)
await socialAPI.unfollow(userId)
const followers = await socialAPI.getFollowers(userId)

// Feed operations
const feed = await socialAPI.getFeed(limit, offset)
const explore = await socialAPI.getExplorePosts(limit, offset)

// Post operations
await socialAPI.createPost({ type: 'status', content: '...' })
await socialAPI.likePost(postId)
await socialAPI.addComment(postId, content)

// Referral operations
const stats = await socialAPI.getReferralStats()
const referrals = await socialAPI.getReferrals()
await socialAPI.generateReferralCode()

// Search operations
const users = await socialAPI.searchUsers(query)
const posts = await socialAPI.searchPosts(query)
```

### State Management

Uses **React Query** for server state:

```typescript
// Example: Fetch feed
const { data: posts, isLoading } = useQuery({
  queryKey: ['feed', activeTab],
  queryFn: () => socialAPI.getFeed(),
})

// Example: Like mutation
const likeMutation = useMutation({
  mutationFn: ({ postId, isLiked }) =>
    isLiked ? socialAPI.unlikePost(postId) : socialAPI.likePost(postId),
  onSuccess: () => {
    queryClient.invalidateQueries({ queryKey: ['feed'] })
  },
})
```

### Data Types

```typescript
interface UserProfile {
  id: string
  username: string
  bio?: string
  location?: string
  website?: string
  github?: string
  followers: number
  following: number
  isFollowing?: boolean
  reputation: number
  rank: number
  // ...
}

interface Post {
  id: string
  userId: string
  username: string
  type: 'submission' | 'job' | 'achievement' | 'status'
  content: string
  code?: string
  language?: string
  likes: number
  comments: number
  shares: number
  isLiked?: boolean
  timestamp: number
  // ...
}

interface Referral {
  id: string
  code: string
  referrerId: string
  referredUserId?: string
  status: 'pending' | 'completed'
  reward: number
  createdAt: number
  completedAt?: number
}
```

---

## 🎨 UI Components

### Post Card
Reusable component for displaying posts in feed:

```tsx
<PostCard
  post={post}
  index={index}
  onLike={(postId, isLiked) => handleLike(postId, isLiked)}
  onUserClick={(userId) => navigate(`/users/${userId}`)}
/>
```

Features:
- User avatar and username (clickable → profile)
- Post content with formatting
- Code editor for code snippets
- Like/comment/share buttons
- Score badge for submissions
- Payout display for earnings

### User Card
Compact user display for follower/following lists:

```tsx
<UserCard user={user} />
```

Features:
- Avatar with gradient
- Username and rank
- Reputation score
- "View Profile" button

---

## 🚀 Navigation Updates

### New Routes:
```typescript
// Social routes
<Route path="feed" element={<FeedPage />} />
<Route path="users/:userId" element={<UserProfilePage />} />
<Route path="referrals" element={<ReferralPage />} />
```

### Updated Navigation Bar:
- **Home** - Landing page
- **Feed** 📈 - Social activity feed (NEW)
- **Jobs** - Browse/submit jobs
- **Leaderboard** - Rankings
- **Refer & Earn** 🎁 - Referral dashboard (NEW)

---

## 💰 Referral Economics

### Earning Calculation:
```
Base Referral Reward: 100 JC
Friend Signup Bonus: 50 JC
Trigger: Friend completes first job

Milestone Bonuses:
- 5 referrals: +500 JC
- 10 referrals: +1,200 JC
- 25 referrals: +3,500 JC + 10% earnings boost
```

### Example Scenario:
```
You refer 10 friends:
- Base earnings: 10 × 100 JC = 1,000 JC
- 5-referral bonus: +500 JC
- 10-referral bonus: +1,200 JC
- Total: 2,700 JC + "Community Builder" badge
```

### Fraud Prevention:
- IP address tracking
- Device fingerprinting
- Email verification
- Minimum account age before referral eligibility
- Suspicious activity detection

---

## 📊 Analytics & Tracking

### User Stats:
- Follower growth over time
- Post engagement rate (likes per post)
- Most popular posts
- Referral conversion rate

### Platform Stats:
- Total users
- Active users (last 30 days)
- Total referrals
- Referral success rate
- Average referrals per user

---

## 🔒 Privacy & Moderation

### Profile Privacy:
- Public profiles by default
- Option to hide balance/earnings (future)
- Block users (future)
- Private accounts (future)

### Content Moderation:
- Report posts/comments
- Automated spam detection
- Community guidelines enforcement
- Moderator tools (future)

---

## 🎯 Gamification Elements

### Social Achievements:
- **First Follower**: Get your first follower (+10 JC)
- **Popular Post**: Get 100 likes on a single post (+50 JC)
- **Influencer**: Reach 100 followers (+200 JC)
- **Community Leader**: Reach 1,000 followers (+1,000 JC)
- **Code Showcase Master**: Share 50 code snippets (+500 JC)

### Referral Badges:
- 🌱 **Starter** (1-4 referrals)
- ⭐ **Influencer** (5-9 referrals)
- 🏆 **Community Builder** (10-24 referrals)
- 👑 **Ambassador** (25+ referrals)

---

## 🔮 Future Enhancements

### Phase 1 (Current):
- ✅ Social feed
- ✅ Follow system
- ✅ Referral rewards
- ✅ User profiles
- ✅ Basic interactions (like/comment)

### Phase 2 (Planned):
- [ ] Real-time notifications
- [ ] Direct messaging
- [ ] Code collaboration features
- [ ] Team competitions
- [ ] Hackathon events

### Phase 3 (Future):
- [ ] Live coding streams
- [ ] Video uploads
- [ ] Marketplace for code snippets
- [ ] Premium subscriptions
- [ ] Custom profile themes

---

## 📈 Success Metrics

### Key Performance Indicators (KPIs):
1. **Daily Active Users (DAU)**
2. **Posts per User per Day**
3. **Referral Conversion Rate**
4. **Average Follower Count**
5. **Engagement Rate** (likes + comments / posts)
6. **Retention Rate** (7-day, 30-day)

### Target Goals:
- 50% of users share at least 1 post per week
- 30% referral conversion rate
- Average 10+ followers per active user
- 80% 7-day retention rate

---

## 🛠️ Development Checklist

### Frontend (Complete):
- [x] Social API client (`social.ts`)
- [x] Feed page with post creation
- [x] Referral dashboard
- [x] User profile pages
- [x] Follow/unfollow functionality
- [x] Like/comment interactions
- [x] Navigation updates

### Backend (Required):
- [ ] Social endpoints in Janus Service Gateway
- [ ] Database schema for posts, follows, referrals
- [ ] Referral code generation
- [ ] Referral tracking and payout logic
- [ ] Feed algorithm (chronological + relevance)
- [ ] Search functionality
- [ ] Content moderation tools

### Infrastructure:
- [ ] WebSocket for real-time updates
- [ ] Image/video upload support
- [ ] CDN for media files
- [ ] Redis for caching feed data
- [ ] Elasticsearch for search

---

## 🚦 Getting Started

### For Users:
1. **Join the Platform**: Sign up with a referral code to get 50 JC bonus
2. **Complete Your Profile**: Add bio, links, avatar
3. **Follow Developers**: Find interesting people to follow
4. **Share Your Work**: Post code snippets and achievements
5. **Invite Friends**: Use your referral link to earn JC

### For Developers:
1. **Read the API docs**: `src/api/social.ts`
2. **Understand data flow**: React Query + Zustand
3. **Test locally**: Mock data available in API client
4. **Add backend**: Implement endpoints in Service Gateway
5. **Deploy**: Build and push to MeshISP

---

## 📞 Support

Questions about social features?

- **GitHub**: [@Thoseidiots](https://github.com/Thoseidiots)
- **Email**: legac3y@gmail.com
- **Mesh**: `jmaxing.mesh:3001`

---

**J-MAXING Social Features v1.0**
*Code together. Grow together. Earn together.*
