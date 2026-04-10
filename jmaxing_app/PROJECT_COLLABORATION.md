# J-MAXING Project Collaboration System

Complete documentation for the project collaboration features (Community, Group, and Solo projects).

---

## 🌟 Overview

J-MAXING now supports **collaborative coding projects** with three distinct modes:

1. **Community Projects** - Open collaboration, anyone can join
2. **Group Projects** - Invite-only teams with private chat
3. **Solo Projects** - Individual work with optional showcase

---

## 📊 Project Types Comparison

| Feature | Community | Group | Solo |
|---------|-----------|-------|------|
| **Visibility** | Public | Public/Private | Public/Private |
| **Join Method** | Anyone can join | Invite-only | N/A (one person) |
| **Chat** | Public group chat | Private group chat | N/A |
| **Rewards** | Shared reward pool | Optional rewards | Personal tracking |
| **Contributors** | Unlimited | Limited by invites | 1 (you) |
| **Best For** | Open-source, learning | Team projects | Personal portfolio |

---

## 🚀 Features

### 1. Project Management

#### Create Project
- Choose type (Community/Group/Solo)
- Set name, description, tags
- Link GitHub repository
- Specify language and difficulty
- Set optional deadline

#### Project Settings
- Public/Private visibility
- Allow auto-join (Community only)
- Reward pool configuration
- Member management

#### Project Stats
- Total members
- Total contributions
- Lines of code added/removed
- Earnings distributed
- Activity timeline

---

### 2. Community Projects 🌐

**Best for**: Open-source projects, learning together, community building

#### Features:
- **Open Join** - Anyone can join and contribute immediately
- **Public Chat** - Collaborate in real-time with all contributors
- **Reward Pool** - Owner deposits JC, distributed based on contributions
- **Contribution Tracking** - All commits, PRs, issues tracked automatically
- **Leaderboard** - Top contributors get recognized

#### Reward Distribution Methods:

**Equal Distribution**:
```
Reward Pool: 1,000 JC
Contributors: 5
Each gets: 200 JC
```

**Contribution-Based** (recommended):
```
Total Lines Changed: 1,000
- Alice: 500 lines → 50% → 500 JC
- Bob: 300 lines → 30% → 300 JC
- Charlie: 200 lines → 20% → 200 JC
```

**Merit-Based**:
- Owner manually approves contributions
- Can set custom rewards per contribution
- Quality > Quantity

#### Example Use Cases:
- Open-source library development
- Educational coding challenges
- Hackathon projects
- Community-driven tools

---

### 3. Group Projects 🔒

**Best for**: Team projects, client work, private collaboration

#### Features:
- **Invite-Only** - Control who joins your team
- **Private Chat** - Team communication stays within group
- **Role Management** - Owner, Admin, Member, Viewer roles
- **Invite System** - Send invites, pending/accepted/declined tracking
- **Optional Rewards** - Split earnings among team members

#### Roles & Permissions:

| Role | View | Contribute | Invite | Manage | Delete |
|------|------|------------|--------|--------|--------|
| Owner | ✅ | ✅ | ✅ | ✅ | ✅ |
| Admin | ✅ | ✅ | ✅ | ✅ | ❌ |
| Member | ✅ | ✅ | ❌ | ❌ | ❌ |
| Viewer | ✅ | ❌ | ❌ | ❌ | ❌ |

#### Invite Flow:
```
1. Owner/Admin invites user (specify role)
2. User receives notification
3. User accepts/declines invite
4. If accepted, user joins project with specified role
5. Team is notified in group chat
```

#### Example Use Cases:
- Startup development teams
- Client projects
- Research collaborations
- Study group assignments

---

### 4. Solo Projects ⭐

**Best for**: Personal learning, portfolio projects, skill building

#### Features:
- **Personal Tracking** - Track your own progress
- **Optional Showcase** - Share finished work on your profile
- **Milestone System** - Set personal goals
- **Time Tracking** - See hours invested
- **Export Stats** - Download contribution data

#### Why Use Solo Projects?
- **Learning** - Practice without pressure
- **Portfolio** - Build showcase-able projects
- **Experimentation** - Try new technologies
- **Privacy** - Work on ideas before sharing

#### Stats Tracked:
- Commits over time
- Lines of code written
- Languages used
- Time spent coding
- Milestones achieved

#### Example Use Cases:
- Learning a new framework
- Building personal portfolio site
- Coding challenges (LeetCode style)
- Private side projects

---

## 💬 Group Chat System

Available in Community and Group projects (if you're a member).

### Features:
- **Real-time messaging** - Updates every 3 seconds
- **Message history** - Full chat log preserved
- **Reply threading** - Reply to specific messages
- **Emoji reactions** - React with emojis
- **Code snippets** - Share code directly in chat
- **Attachments** - Upload files/images
- **Mentions** - @username notifications
- **Edit/Delete** - Edit or remove your messages

### Chat Commands (Future):
```
/code python
  # Share code snippet with syntax highlighting

/invite @username
  # Invite user to project

/stats
  # Show project stats in chat

/poll "Question" "Option1" "Option2"
  # Create a poll for team decisions
```

---

## 🎯 Contribution System

Track all code contributions automatically.

### Contribution Types:
1. **Commit** - Direct commits to repository
2. **Pull Request** - PRs submitted for review
3. **Issue** - Bugs/features reported
4. **Comment** - Code review comments
5. **Review** - Pull request reviews

### Tracked Metrics:
- Lines added (green)
- Lines removed (red)
- Files changed
- Commits count
- Review comments
- Issues resolved

### Approval Flow (Community Projects):
```
1. Contributor submits code
2. Owner/Admin reviews contribution
3. Approve → Contributor earns JC
4. Reject → No payout, feedback provided
```

---

## 🏆 Rewards & Earnings

### For Community Projects:

**Setup**:
1. Owner creates project
2. Sets reward pool (e.g., 5,000 JC)
3. Chooses distribution method
4. Contributions start coming in

**Distribution**:
- Automatic based on chosen method
- Payouts when project completes or milestone reached
- Top contributors get bonus rewards

**Example**:
```
Project: Build a Todo App
Reward Pool: 5,000 JC
Distribution: Contribution-based
Deadline: 30 days

After 30 days:
- Total contributions: 50 (from 10 people)
- Rewards distributed based on quality + quantity
- Top 3 contributors get 2x multiplier
```

### For Group Projects:

**Optional Rewards**:
- Owner can set rewards for specific tasks
- Manual approval required
- Can use external JC (not from pool)

---

## 📱 User Interface

### Projects Page (`/projects`)

**Tabs**:
- All Projects
- My Projects
- Community
- Group
- Solo

**Filters**:
- Search by name/tags
- Filter by language
- Filter by difficulty
- Sort by popularity/newest/reward

**Project Cards Show**:
- Project name and description
- Owner username
- Type badge (Community/Group/Solo)
- Member count
- Contribution count
- Star count
- Reward pool (if any)
- Tags and language

---

### Project Detail Page (`/projects/:id`)

**Tabs**:
1. **Overview** - Project details, repo link, stats
2. **Members** - All contributors with roles
3. **Contributions** - Recent commits/PRs/issues
4. **Chat** - Real-time group chat

**Header Shows**:
- Project name and description
- Type and status badges
- Star button
- Join/Leave/Invite buttons
- Settings (owner only)

**Stats Grid**:
- Members count
- Contributions count
- Primary language
- Reward pool (if any)

---

## 🔧 API Reference

### Projects API
**File**: `src/api/projects.ts`

```typescript
// List projects
projectsAPI.listProjects({
  type?: 'community' | 'group' | 'solo',
  status?: 'active' | 'completed' | 'archived',
  tags?: string[],
  language?: string,
  search?: string,
})

// Get project details
projectsAPI.getProject(projectId)

// Create project
projectsAPI.createProject({
  name: string,
  description: string,
  type: ProjectType,
  repository?: string,
  tags: string[],
  language: string,
  difficulty: number,
  rewardPool?: number,
  rewardDistribution?: 'equal' | 'contribution-based' | 'merit-based',
})

// Join community project
projectsAPI.joinProject(projectId)

// Leave project
projectsAPI.leaveProject(projectId)

// Members
projectsAPI.getMembers(projectId)
projectsAPI.inviteMember(projectId, userId, role)
projectsAPI.removeMember(projectId, userId)

// Contributions
projectsAPI.getContributions(projectId)
projectsAPI.submitContribution(projectId, {...})
projectsAPI.approveContribution(projectId, contributionId, reward)

// Chat
projectsAPI.getMessages(projectId, limit, offset)
projectsAPI.sendMessage(projectId, content, replyTo?)
projectsAPI.deleteMessage(projectId, messageId)
projectsAPI.reactToMessage(projectId, messageId, emoji)

// Stats
projectsAPI.getStats(projectId)

// Star/Unstar
projectsAPI.starProject(projectId)
projectsAPI.unstarProject(projectId)
```

---

## 🎨 UI Components

### ProjectCard
Reusable card component for project listings:

```tsx
<ProjectCard project={project} index={index} />
```

Shows:
- Type icon (Users/Lock/Star)
- Name and owner
- Description (truncated)
- Tags (first 3)
- Stats (members, contributions, stars)
- Status and language badges
- Reward pool (if any)

### Chat Interface
Real-time chat with polling:

```tsx
<ChatInterface projectId={projectId} />
```

Features:
- Message list with avatars
- Input field with Send button
- Auto-scroll to latest
- 3-second polling for new messages

---

## 🚀 Getting Started

### Create a Community Project:

1. Go to `/projects`
2. Click "Create Project"
3. Select "Community" type
4. Fill in project details
5. Set reward pool (optional)
6. Choose distribution method
7. Click "Create Project"
8. Share project link to attract contributors

### Join a Community Project:

1. Browse `/projects`
2. Find interesting project
3. Click to view details
4. Click "Join Project"
5. Start contributing immediately
6. Chat with team in group chat

### Create a Group Project:

1. Go to `/projects`
2. Click "Create Project"
3. Select "Group" type
4. Fill in project details
5. Click "Create Project"
6. Invite team members via Settings
7. Collaborate in private chat

### Start a Solo Project:

1. Go to `/projects`
2. Click "Create Project"
3. Select "Solo" type
4. Fill in project details
5. Click "Create Project"
6. Track your progress personally

---

## 📊 Success Metrics

### For Platform:
- Total projects created
- Active projects (contributions in last 7 days)
- Average members per community project
- Total JC distributed via projects
- Project completion rate

### For Users:
- Projects joined
- Contributions made
- JC earned from projects
- Reputation gained
- Team connections made

---

## 🔒 Security & Moderation

### Access Control:
- Type-based permissions (Community/Group/Solo)
- Role-based actions (Owner/Admin/Member/Viewer)
- Private projects hidden from non-members
- Invite-only enforcement for group projects

### Content Moderation:
- Report projects/contributions/messages
- Spam detection on project creation
- Abuse detection in chat
- Owner can remove malicious members

---

## 🔮 Future Enhancements

### Phase 1 (Current):
- ✅ Three project types
- ✅ Group chat
- ✅ Contribution tracking
- ✅ Reward distribution
- ✅ Invite system

### Phase 2 (Planned):
- [ ] GitHub integration (auto-sync commits)
- [ ] Code review UI within J-MAXING
- [ ] Project templates
- [ ] Milestone tracking
- [ ] Gantt charts / Project boards

### Phase 3 (Future):
- [ ] Video calls for group projects
- [ ] Screen sharing
- [ ] Live coding sessions
- [ ] Project forking
- [ ] Cross-project collaboration

---

## 🎯 Use Case Examples

### Example 1: Open-Source Library
```
Type: Community
Reward Pool: 10,000 JC
Distribution: Contribution-based
Members: 25
Goal: Build a React component library

Result:
- 500+ commits
- 15 active contributors
- Library published to npm
- Top 3 contributors earned 3,000+ JC each
```

### Example 2: Startup MVP
```
Type: Group
Members: 4 (co-founders)
Private: Yes
Rewards: Equity-based (external to JC)
Goal: Build MVP in 3 months

Result:
- Private collaboration
- Daily standups in chat
- 1,000+ commits
- MVP completed on time
```

### Example 3: Learning Journey
```
Type: Solo
Owner: You
Goal: Learn Rust by building a CLI tool
Duration: 2 months

Result:
- 200+ commits tracked
- 5,000 lines of Rust written
- Portfolio piece completed
- Skills improved, stats tracked
```

---

## 📞 Support

Questions about project collaboration?

- **GitHub**: [@Thoseidiots](https://github.com/Thoseidiots)
- **Email**: legac3y@gmail.com
- **Mesh**: `jmaxing.mesh:3001`

---

**J-MAXING Project Collaboration v1.0**
*Build together. Learn together. Earn together.*
