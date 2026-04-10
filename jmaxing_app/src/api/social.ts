import { apiClient } from './client'

// Social API Types
export interface UserProfile {
  id: string
  username: string
  email: string
  avatar?: string
  bio?: string
  location?: string
  website?: string
  github?: string
  twitter?: string
  balance: number
  reputation: number
  rank: number
  tasksCompleted: number
  joinedAt: number
  followers: number
  following: number
  isFollowing?: boolean
}

export interface Post {
  id: string
  userId: string
  username: string
  avatar?: string
  type: 'submission' | 'job' | 'achievement' | 'status'
  content: string
  code?: string
  language?: string
  jobId?: string
  submissionId?: string
  score?: number
  payout?: number
  likes: number
  comments: number
  shares: number
  timestamp: number
  isLiked?: boolean
  images?: string[]
  tags?: string[]
}

export interface Comment {
  id: string
  postId: string
  userId: string
  username: string
  avatar?: string
  content: string
  likes: number
  timestamp: number
  isLiked?: boolean
  replies?: Comment[]
}

export interface ActivityItem {
  id: string
  userId: string
  username: string
  avatar?: string
  type: 'follow' | 'like' | 'comment' | 'submission' | 'job_created' | 'achievement'
  description: string
  timestamp: number
  metadata?: {
    postId?: string
    jobId?: string
    submissionId?: string
    achievementName?: string
  }
}

export interface Referral {
  id: string
  code: string
  referrerId: string
  referredUserId?: string
  referredUsername?: string
  status: 'pending' | 'completed'
  reward: number
  createdAt: number
  completedAt?: number
}

export interface ReferralStats {
  totalReferrals: number
  completedReferrals: number
  pendingReferrals: number
  totalEarned: number
  referralCode: string
  referralLink: string
}

// Social API Functions
export const socialAPI = {
  // User Profile
  getProfile: async (userId: string) => {
    const response = await apiClient.get<UserProfile>(`/users/${userId}`)
    return response.data
  },

  updateProfile: async (updates: Partial<UserProfile>) => {
    const response = await apiClient.put('/users/me', updates)
    return response.data
  },

  // Follow System
  follow: async (userId: string) => {
    const response = await apiClient.post(`/users/${userId}/follow`)
    return response.data
  },

  unfollow: async (userId: string) => {
    const response = await apiClient.delete(`/users/${userId}/follow`)
    return response.data
  },

  getFollowers: async (userId: string) => {
    const response = await apiClient.get<UserProfile[]>(`/users/${userId}/followers`)
    return response.data
  },

  getFollowing: async (userId: string) => {
    const response = await apiClient.get<UserProfile[]>(`/users/${userId}/following`)
    return response.data
  },

  // Feed
  getFeed: async (limit: number = 20, offset: number = 0) => {
    const response = await apiClient.get<Post[]>('/feed', {
      params: { limit, offset },
    })
    return response.data
  },

  getExplorePosts: async (limit: number = 20, offset: number = 0) => {
    const response = await apiClient.get<Post[]>('/explore', {
      params: { limit, offset },
    })
    return response.data
  },

  getUserPosts: async (userId: string, limit: number = 20) => {
    const response = await apiClient.get<Post[]>(`/users/${userId}/posts`, {
      params: { limit },
    })
    return response.data
  },

  // Posts
  createPost: async (post: {
    type: 'status' | 'showcase'
    content: string
    code?: string
    language?: string
    images?: string[]
    tags?: string[]
  }) => {
    const response = await apiClient.post('/posts', post)
    return response.data
  },

  deletePost: async (postId: string) => {
    const response = await apiClient.delete(`/posts/${postId}`)
    return response.data
  },

  // Likes
  likePost: async (postId: string) => {
    const response = await apiClient.post(`/posts/${postId}/like`)
    return response.data
  },

  unlikePost: async (postId: string) => {
    const response = await apiClient.delete(`/posts/${postId}/like`)
    return response.data
  },

  // Comments
  getComments: async (postId: string) => {
    const response = await apiClient.get<Comment[]>(`/posts/${postId}/comments`)
    return response.data
  },

  addComment: async (postId: string, content: string) => {
    const response = await apiClient.post(`/posts/${postId}/comments`, {
      content,
    })
    return response.data
  },

  deleteComment: async (postId: string, commentId: string) => {
    const response = await apiClient.delete(`/posts/${postId}/comments/${commentId}`)
    return response.data
  },

  likeComment: async (postId: string, commentId: string) => {
    const response = await apiClient.post(
      `/posts/${postId}/comments/${commentId}/like`
    )
    return response.data
  },

  // Activity
  getActivity: async (limit: number = 50) => {
    const response = await apiClient.get<ActivityItem[]>('/activity', {
      params: { limit },
    })
    return response.data
  },

  // Referrals
  getReferralStats: async () => {
    const response = await apiClient.get<ReferralStats>('/referrals/stats')
    return response.data
  },

  getReferrals: async () => {
    const response = await apiClient.get<Referral[]>('/referrals')
    return response.data
  },

  generateReferralCode: async () => {
    const response = await apiClient.post('/referrals/generate')
    return response.data
  },

  applyReferralCode: async (code: string) => {
    const response = await apiClient.post('/referrals/apply', { code })
    return response.data
  },

  // Search
  searchUsers: async (query: string, limit: number = 20) => {
    const response = await apiClient.get<UserProfile[]>('/users/search', {
      params: { q: query, limit },
    })
    return response.data
  },

  searchPosts: async (query: string, limit: number = 20) => {
    const response = await apiClient.get<Post[]>('/posts/search', {
      params: { q: query, limit },
    })
    return response.data
  },
}
