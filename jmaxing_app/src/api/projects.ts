import { apiClient } from './client'

// Project Types
export type ProjectType = 'community' | 'group' | 'solo'
export type ProjectStatus = 'active' | 'completed' | 'archived'
export type MemberRole = 'owner' | 'admin' | 'member' | 'viewer'

export interface Project {
  id: string
  name: string
  description: string
  type: ProjectType
  status: ProjectStatus
  ownerId: string
  ownerUsername: string

  // Metadata
  repository?: string  // GitHub repo URL
  tags: string[]
  language: string     // Primary language
  difficulty: number   // 1-5

  // Stats
  members: number
  contributions: number
  stars: number

  // Rewards (for community projects)
  rewardPool?: number  // Total JC available
  rewardDistribution?: 'equal' | 'contribution-based' | 'merit-based'

  // Settings
  isPublic: boolean
  allowJoin: boolean   // Can anyone join (community) or invite-only (group)

  // Timestamps
  createdAt: number
  updatedAt: number
  deadline?: number

  // User-specific
  isMember?: boolean
  userRole?: MemberRole
}

export interface ProjectMember {
  id: string
  projectId: string
  userId: string
  username: string
  avatar?: string
  role: MemberRole
  contributions: number
  linesAdded: number
  linesRemoved: number
  commits: number
  earnedJC: number
  joinedAt: number
}

export interface Contribution {
  id: string
  projectId: string
  userId: string
  username: string
  type: 'commit' | 'pull_request' | 'issue' | 'comment' | 'review'
  title: string
  description: string
  code?: string
  filesChanged?: string[]
  linesAdded: number
  linesRemoved: number
  reward?: number      // JC earned for this contribution
  status: 'pending' | 'approved' | 'rejected'
  timestamp: number
}

export interface ProjectInvite {
  id: string
  projectId: string
  projectName: string
  inviterId: string
  inviterUsername: string
  inviteeId: string
  inviteeUsername: string
  role: MemberRole
  status: 'pending' | 'accepted' | 'declined'
  createdAt: number
  expiresAt: number
}

export interface ChatMessage {
  id: string
  projectId: string
  userId: string
  username: string
  avatar?: string
  content: string
  attachments?: string[]
  replyTo?: string     // Message ID
  timestamp: number
  edited?: boolean
  reactions?: { emoji: string; userIds: string[] }[]
}

export interface ProjectStats {
  totalContributions: number
  totalMembers: number
  totalCommits: number
  totalLinesAdded: number
  totalLinesRemoved: number
  totalEarned: number
  topContributors: {
    userId: string
    username: string
    contributions: number
    earnedJC: number
  }[]
  activityTimeline: {
    date: string
    contributions: number
    commits: number
  }[]
}

// Projects API
export const projectsAPI = {
  // List projects
  listProjects: async (filters?: {
    type?: ProjectType
    status?: ProjectStatus
    tags?: string[]
    language?: string
    search?: string
    limit?: number
    offset?: number
  }) => {
    const response = await apiClient.get<Project[]>('/projects', {
      params: filters,
    })
    return response.data
  },

  // Get my projects
  getMyProjects: async (type?: ProjectType) => {
    const response = await apiClient.get<Project[]>('/projects/me', {
      params: { type },
    })
    return response.data
  },

  // Get project details
  getProject: async (projectId: string) => {
    const response = await apiClient.get<Project>(`/projects/${projectId}`)
    return response.data
  },

  // Create project
  createProject: async (project: {
    name: string
    description: string
    type: ProjectType
    repository?: string
    tags: string[]
    language: string
    difficulty: number
    rewardPool?: number
    rewardDistribution?: 'equal' | 'contribution-based' | 'merit-based'
    isPublic?: boolean
    allowJoin?: boolean
    deadline?: number
  }) => {
    const response = await apiClient.post('/projects', project)
    return response.data
  },

  // Update project
  updateProject: async (projectId: string, updates: Partial<Project>) => {
    const response = await apiClient.put(`/projects/${projectId}`, updates)
    return response.data
  },

  // Delete project
  deleteProject: async (projectId: string) => {
    const response = await apiClient.delete(`/projects/${projectId}`)
    return response.data
  },

  // Members
  getMembers: async (projectId: string) => {
    const response = await apiClient.get<ProjectMember[]>(
      `/projects/${projectId}/members`
    )
    return response.data
  },

  // Join project (community)
  joinProject: async (projectId: string) => {
    const response = await apiClient.post(`/projects/${projectId}/join`)
    return response.data
  },

  // Leave project
  leaveProject: async (projectId: string) => {
    const response = await apiClient.post(`/projects/${projectId}/leave`)
    return response.data
  },

  // Invite to project (group)
  inviteMember: async (projectId: string, userId: string, role: MemberRole) => {
    const response = await apiClient.post(`/projects/${projectId}/invite`, {
      userId,
      role,
    })
    return response.data
  },

  // Get invites
  getInvites: async () => {
    const response = await apiClient.get<ProjectInvite[]>('/projects/invites')
    return response.data
  },

  // Respond to invite
  respondToInvite: async (inviteId: string, accept: boolean) => {
    const response = await apiClient.post(`/projects/invites/${inviteId}/respond`, {
      accept,
    })
    return response.data
  },

  // Update member role
  updateMemberRole: async (
    projectId: string,
    userId: string,
    role: MemberRole
  ) => {
    const response = await apiClient.put(
      `/projects/${projectId}/members/${userId}`,
      { role }
    )
    return response.data
  },

  // Remove member
  removeMember: async (projectId: string, userId: string) => {
    const response = await apiClient.delete(
      `/projects/${projectId}/members/${userId}`
    )
    return response.data
  },

  // Contributions
  getContributions: async (projectId: string, limit: number = 50) => {
    const response = await apiClient.get<Contribution[]>(
      `/projects/${projectId}/contributions`,
      { params: { limit } }
    )
    return response.data
  },

  // Submit contribution
  submitContribution: async (
    projectId: string,
    contribution: {
      type: 'commit' | 'pull_request' | 'issue' | 'comment' | 'review'
      title: string
      description: string
      code?: string
      filesChanged?: string[]
      linesAdded: number
      linesRemoved: number
    }
  ) => {
    const response = await apiClient.post(
      `/projects/${projectId}/contributions`,
      contribution
    )
    return response.data
  },

  // Approve contribution (admin only)
  approveContribution: async (
    projectId: string,
    contributionId: string,
    reward?: number
  ) => {
    const response = await apiClient.post(
      `/projects/${projectId}/contributions/${contributionId}/approve`,
      { reward }
    )
    return response.data
  },

  // Chat
  getMessages: async (projectId: string, limit: number = 100, offset: number = 0) => {
    const response = await apiClient.get<ChatMessage[]>(
      `/projects/${projectId}/chat`,
      { params: { limit, offset } }
    )
    return response.data
  },

  // Send message
  sendMessage: async (
    projectId: string,
    content: string,
    replyTo?: string
  ) => {
    const response = await apiClient.post(`/projects/${projectId}/chat`, {
      content,
      replyTo,
    })
    return response.data
  },

  // Delete message
  deleteMessage: async (projectId: string, messageId: string) => {
    const response = await apiClient.delete(
      `/projects/${projectId}/chat/${messageId}`
    )
    return response.data
  },

  // React to message
  reactToMessage: async (
    projectId: string,
    messageId: string,
    emoji: string
  ) => {
    const response = await apiClient.post(
      `/projects/${projectId}/chat/${messageId}/react`,
      { emoji }
    )
    return response.data
  },

  // Stats
  getStats: async (projectId: string) => {
    const response = await apiClient.get<ProjectStats>(
      `/projects/${projectId}/stats`
    )
    return response.data
  },

  // Star project
  starProject: async (projectId: string) => {
    const response = await apiClient.post(`/projects/${projectId}/star`)
    return response.data
  },

  // Unstar project
  unstarProject: async (projectId: string) => {
    const response = await apiClient.delete(`/projects/${projectId}/star`)
    return response.data
  },

  // Search projects
  searchProjects: async (query: string, limit: number = 20) => {
    const response = await apiClient.get<Project[]>('/projects/search', {
      params: { q: query, limit },
    })
    return response.data
  },
}
